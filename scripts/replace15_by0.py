from pathlib import Path

# 🔧 CHANGE THIS PATH IF NEEDED
LABELS_ROOT = Path(
    "dataset/usa/golden_data/labels"
)

assert LABELS_ROOT.exists(), f"Path not found: {LABELS_ROOT}"

fixed_files = 0
fixed_lines = 0
skipped_empty = 0

for label_file in LABELS_ROOT.rglob("*.txt"):
    content = label_file.read_text().strip()

    # 🔹 Case 1: empty label file (no objects) → leave untouched
    if content == "":
        skipped_empty += 1
        continue

    new_lines = []
    changed = False

    for line in content.splitlines():
        parts = line.strip().split()

        # 🔹 Case 2: malformed line → skip safely
        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts

        # 🔹 Replace class 15 → 0
        if cls == "15":
            cls = "0"
            changed = True
            fixed_lines += 1

        new_lines.append(f"{cls} {x} {y} {w} {h}")

    if changed:
        label_file.write_text("\n".join(new_lines))
        fixed_files += 1

print("\n✅ YOLO label fixing completed")
print(f"   📝 Files modified        : {fixed_files}")
print(f"   🔁 Labels remapped (15→0): {fixed_lines}")
print(f"   ⏭️  Empty label files     : {skipped_empty}")
