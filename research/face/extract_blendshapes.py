"""Turn AffectNet face images into 52 MediaPipe blendshape coefficients.

The browser runs the very same FaceLandmarker model (@mediapipe/tasks-vision), so the
features Aura-Face is trained on are the features it sees in deployment. Pixels are
used here, offline, only to produce training features; the deployed model never
receives an image.

Run: python face/extract_blendshapes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET = "Mauregato/affectnet_short"


def main():
    import mediapipe as mp
    from datasets import load_dataset
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HERE / "face_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    ds = load_dataset(DATASET)
    label_names = ds["train"].features["label"].names

    for split in ("train", "val"):
        out = CACHE / f"affectnet_blendshapes_{split}.npz"
        if out.exists():
            print(f"{split}: cached")
            continue
        d = ds[split]
        feats = np.zeros((len(d), 52), np.float32)
        found = np.zeros(len(d), bool)
        names = None
        t0 = time.time()
        for i, row in enumerate(d):
            img = np.ascontiguousarray(np.asarray(row["image"].convert("RGB")))
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
            if res.face_blendshapes:
                cats = res.face_blendshapes[0]
                feats[i] = [c.score for c in cats]
                names = names or [c.category_name for c in cats]
                found[i] = True
            if (i + 1) % 2000 == 0:
                print(f"  {split} {i + 1}/{len(d)} detected={found[:i + 1].mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
        np.savez_compressed(out, x=feats, y=np.array(d["label"]), found=found,
                            names=np.array(names), labels=np.array(label_names))
        print(f"{split}: {found.sum()}/{len(d)} faces -> {out}")


if __name__ == "__main__":
    main()
