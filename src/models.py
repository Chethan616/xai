"""Model zoo: the two baselines and the candidate explainable architectures.
All take (B, 1, L) z-scored signals; MODELS[name](num_classes) builds one.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CYCLE = 64  # samples per cycle in PQD-16c (used by the multiview channels)


# ------------------------------------------------------------------ baselines

class ResidualConvBlock(nn.Module):
    def __init__(self, cin, cout, k=5, dilation=1):
        super().__init__()
        p = ((k - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(cin, cout, k, padding=p, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(cout, cout, k, padding=p, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
                     if cin != cout else nn.Identity())
        self.activation = nn.GELU()

    def forward(self, x):
        idt = self.skip(x)
        x = self.act1(self.bn1(self.conv1(x)))
        return self.activation(self.bn2(self.conv2(x)) + idt)


class CNNResBiLSTM(nn.Module):
    """Baseline 1 — identical to models/legacy/cnn_res_bilstm_pq.pth."""

    def __init__(self, num_classes=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.GELU(),
            ResidualConvBlock(32, 32, 5, 1), nn.MaxPool1d(2),
            ResidualConvBlock(32, 64, 5, 1), nn.MaxPool1d(2),
            ResidualConvBlock(64, 128, 5, 2), nn.MaxPool1d(2),
            ResidualConvBlock(128, 128, 3, 2),
        )
        self.sequence_norm = nn.LayerNorm(128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.30),
                                        nn.Linear(128, num_classes))

    def forward(self, x, return_attention=False):
        x = self.sequence_norm(self.cnn(x).transpose(1, 2))
        x, _ = self.lstm(x)
        w = torch.softmax(self.attention(x), dim=1)
        out = self.classifier((x * w).sum(1))
        return (out, w.squeeze(-1)) if return_attention else out


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2, bias=False),
                                   nn.BatchNorm1d(cout), nn.GELU())

    def forward(self, x):
        return self.block(x)


class _ComplexRes(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = ConvBNAct(cin, cout, 3)
        self.conv2 = nn.Sequential(nn.Conv1d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm1d(cout))
        self.shortcut = (nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
                         if cin != cout else nn.Identity())

    def forward(self, x):
        return F.gelu(self.conv2(self.conv1(x)) + self.shortcut(x))


class _MultiScale(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.branch3, self.branch5, self.branch7 = (ConvBNAct(cin, c, k) for k in (3, 5, 7))
        self.fusion = nn.Sequential(nn.Conv1d(3 * c, 3 * c, 1, bias=False), nn.BatchNorm1d(3 * c), nn.GELU())

    def forward(self, x):
        return self.fusion(torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], 1))


class CNNComplex(nn.Module):
    """Baseline 2 — rebuilt from the layer shapes in models/legacy/cnn_pq_complex.pth
    (training code was not kept; pooling placement is a best guess)."""

    def __init__(self, num_classes=16):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(1, 32, 7), ConvBNAct(32, 64, 5))
        self.multi_scale = _MultiScale(64, 32)
        self.residual1, self.residual2 = _ComplexRes(96, 128), _ComplexRes(128, 128)
        self.residual3, self.residual4 = _ComplexRes(128, 192), _ComplexRes(192, 192)
        self.deep_features = nn.Sequential(ConvBNAct(192, 256, 5), ConvBNAct(256, 256, 3), ConvBNAct(256, 256, 3))
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.3),
                                        nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        x = F.max_pool1d(self.multi_scale(self.stem(x)), 2)
        x = F.max_pool1d(self.residual2(self.residual1(x)), 2)
        x = F.max_pool1d(self.residual4(self.residual3(x)), 2)
        return self.classifier(self.deep_features(x).mean(-1))


# ------------------------------------------------------------------ candidates

def physics_channels(x):
    """Extra input views computed *inside* the model so attributions stay on the
    raw 640 points: sliding one-cycle RMS (magnitude) and high-pass residual
    (transients / harmonics). x: (B,1,L) normalised signal."""
    rms = torch.sqrt(F.avg_pool1d(F.pad(x ** 2, (CYCLE // 2, CYCLE // 2 - 1), mode="replicate"),
                                  CYCLE, stride=1) + 1e-8)
    # fundamental removed by subtracting the signal one cycle earlier (periodic diff)
    hp = x - torch.roll(x, CYCLE, dims=-1)
    hp[..., :CYCLE] = 0
    return torch.cat([x, rms, hp], 1)


class CAMNet(nn.Module):
    """Fully convolutional ResNet + global average pooling.
    Explainability: Class Activation Map gives per-timestep evidence for every
    class for free (Zhou et al. 2016; Wang et al. 2017 FCN/ResNet for TSC)."""

    def __init__(self, num_classes=16, multiview=False, width=64):
        super().__init__()
        self.multiview = multiview
        cin = 3 if multiview else 1
        self.features = nn.Sequential(
            nn.Conv1d(cin, width, 7, padding=3, bias=False), nn.BatchNorm1d(width), nn.GELU(),
            ResidualConvBlock(width, width, 7), nn.MaxPool1d(2),
            ResidualConvBlock(width, 2 * width, 5), nn.MaxPool1d(2),
            ResidualConvBlock(2 * width, 2 * width, 5, dilation=2),
            ResidualConvBlock(2 * width, 2 * width, 3, dilation=4),
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * width, num_classes)

    def feature_map(self, x):
        return self.features(physics_channels(x) if self.multiview else x)

    def forward(self, x):
        return self.fc(self.dropout(self.feature_map(x).mean(-1)))

    def cam(self, x):
        """(B, C, L) class activation maps upsampled to input length."""
        f = self.feature_map(x)
        m = torch.einsum("kc,bct->bkt", self.fc.weight, f) / f.shape[-1]
        return F.interpolate(m, size=x.shape[-1], mode="linear", align_corners=False)


class _Inception(nn.Module):
    def __init__(self, cin, nf=32, ks=(9, 19, 39)):
        super().__init__()
        self.bottleneck = nn.Conv1d(cin, nf, 1, bias=False) if cin > 1 else nn.Identity()
        b = nf if cin > 1 else cin
        self.convs = nn.ModuleList(nn.Conv1d(b, nf, k, padding=k // 2, bias=False) for k in ks)
        self.pool_conv = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(cin, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)

    def forward(self, x):
        z = self.bottleneck(x)
        return F.relu(self.bn(torch.cat([c(z) for c in self.convs] + [self.pool_conv(x)], 1)))


class InceptionTimeCAM(nn.Module):
    """InceptionTime (Fawaz et al. 2020), depth 6, GAP head -> CAM-explainable."""

    def __init__(self, num_classes=16, multiview=False, nf=32, depth=6):
        super().__init__()
        self.multiview = multiview
        cin = 3 if multiview else 1
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        c, res_c = cin, cin
        for d in range(depth):
            self.blocks.append(_Inception(c, nf))
            c = nf * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(nn.Conv1d(res_c, c, 1, bias=False), nn.BatchNorm1d(c)))
                res_c = c
        self.fc = nn.Linear(c, num_classes)

    def feature_map(self, x):
        x = physics_channels(x) if self.multiview else x
        res = x
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            if d % 3 == 2:
                x = F.relu(x + self.shortcuts[d // 3](res))
                res = x
        return x

    def forward(self, x):
        return self.fc(self.feature_map(x).mean(-1))

    def cam(self, x):
        f = self.feature_map(x)
        m = torch.einsum("kc,bct->bkt", self.fc.weight, f) / f.shape[-1]
        return F.interpolate(m, size=x.shape[-1], mode="linear", align_corners=False)


class CycleTransformer(nn.Module):
    """Half-cycle patches (32 samples -> 20 tokens) + small Transformer.
    Explainability: CLS attention is per half-cycle, a unit an engineer (and an
    LLM) can talk about directly: 'cycles 3-5 drive the decision'."""

    def __init__(self, num_classes=16, d=64, layers=3, heads=4, patch=32, multiview=True):
        super().__init__()
        self.patch, self.multiview = patch, multiview
        cin = 3 if multiview else 1
        self.embed = nn.Sequential(nn.Conv1d(cin, d, 7, padding=3), nn.GELU(),
                                   nn.Conv1d(d, d, patch, stride=patch))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 640 // patch + 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        x = physics_channels(x) if self.multiview else x
        t = self.embed(x).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.fc(self.norm(self.encoder(t)[:, 0]))


MODELS = {
    "cnn_res_bilstm": lambda n=16: CNNResBiLSTM(num_classes=n),             # baseline 1
    "cnn_complex": lambda n=16: CNNComplex(num_classes=n),                  # baseline 2
    "resnet_cam": lambda n=16: CAMNet(num_classes=n),
    "resnet_cam_mv": lambda n=16: CAMNet(num_classes=n, multiview=True),
    "inception_cam": lambda n=16: InceptionTimeCAM(num_classes=n),
    "inception_cam_mv": lambda n=16: InceptionTimeCAM(num_classes=n, multiview=True),
    "cycle_transformer": lambda n=16: CycleTransformer(num_classes=n),
}
