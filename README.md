# pyclnf

Pure Python implementation of OpenFace's CLNF (Constrained Local Neural Fields) facial landmark detector.

## Installation

```bash
pip install pyclnf
```

## Usage

```python
from pyclnf import CLNF

clnf = CLNF()
landmarks, pose = clnf.fit(image)  # 68 facial landmarks + head pose
```

For video, use `clnf.fit()` on consecutive frames—it automatically tracks faces across frames.

## What it does

- Detects 68 facial landmarks
- Estimates 3D head pose (pitch, yaw, roll)
- Uses OpenFace's trained CEN patch experts
- Built-in face detection via [pymtcnn](https://github.com/johnwilsoniv/pymtcnn)

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

CC BY-NC 4.0 — free for non-commercial use with attribution.
