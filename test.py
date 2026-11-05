from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

weights = "runs/detect/train/weights/best.pt"
val_dir = Path("dataset/images/val")
out_dir = Path("result")

images = sorted(
    p for p in val_dir.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

model = YOLO(weights)
results = model.predict(
    source=[str(p) for p in images],
    imgsz=640,
    device=0,
    workers=0,
    verbose=False,
)

out_dir.mkdir(parents=True, exist_ok=True)

for page, start in enumerate(range(0, len(results), 9), start=1):
    chunk = results[start : start + 9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(chunk):
            ax.imshow(chunk[i].plot()[:, :, ::-1])

    out = out_dir / f"Detection_page_{page}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("saved", out)
