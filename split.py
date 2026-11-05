import random
import shutil
from pathlib import Path

images_dir = Path("pylone test/labeled/images")
labels_dir = Path("pylone test/labeled/labels")
out_dir = Path("dataset")
val_ratio = 0.2
seed = 42

exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

pairs = []
for img in sorted(images_dir.iterdir()):
    if img.suffix.lower() not in exts:
        continue
    lbl = labels_dir / f"{img.stem}.txt"
    if lbl.is_file():
        pairs.append((img, lbl))

random.seed(seed)
random.shuffle(pairs)

n_val = max(1, int(len(pairs) * val_ratio))
val_set = pairs[:n_val]
train_set = pairs[n_val:]

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    (out_dir / sub).mkdir(parents=True, exist_ok=True)

for img, lbl in train_set:
    shutil.copy2(img, out_dir / "images" / "train" / img.name)
    shutil.copy2(lbl, out_dir / "labels" / "train" / lbl.name)

for img, lbl in val_set:
    shutil.copy2(img, out_dir / "images" / "val" / img.name)
    shutil.copy2(lbl, out_dir / "labels" / "val" / lbl.name)

yaml = out_dir.resolve().as_posix()
(out_dir / "data.yaml").write_text(
    f"path: {yaml}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: Cell tower\n",
    encoding="utf-8",
)

print("train:", len(train_set))
print("val:", len(val_set))
print("wrote", out_dir / "data.yaml")
