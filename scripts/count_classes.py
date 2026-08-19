import csv
from collections import Counter
from pathlib import Path

CSV_FILE = Path("../dataset/PQD-16c-Data/PQD-16C-Data.csv")
OUTPUT_FILE = Path("class_distribution.txt")

counts = Counter()

with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)

    # Skip header
    next(reader)

    # Read only the first column
    for row in reader:
        if row:
            label = row[0].strip()
            counts[label] += 1


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write("PQD-16C DATASET CLASS DISTRIBUTION\n")
    file.write("=" * 45 + "\n\n")

    for label, count in sorted(counts.items()):
        file.write(f"{label}: {count}\n")

    file.write("\n" + "=" * 45 + "\n")
    file.write(f"Total samples: {sum(counts.values())}\n")
    file.write(f"Total classes: {len(counts)}\n")


print(f"Done. Results written to: {OUTPUT_FILE}")