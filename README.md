# pyclnf

Pure Python implementation of OpenFace's CLNF (Constrained Local Neural Fields) facial landmark detector.

## Installation

```bash
pip install pyclnf
```

pip also installs what pyclnf needs, including OpenCV 4. pyclnf stays on OpenCV 4 on purpose:
OpenCV 5 resamples images slightly differently, which changes the landmarks (see the
[changelog](https://github.com/johnwilsoniv/pyclnf/blob/main/CHANGELOG.md)).

**Optional: PyTorch for speed.** If [PyTorch](https://pytorch.org/get-started/locally/) is
installed, pyclnf uses your graphics processor (Apple Silicon or NVIDIA), which is much faster.
Without PyTorch, pyclnf runs on the processor (CPU). Both work; their landmarks are very close
but not identical, so use the same setup for all the videos you want to compare.

## First run: get OpenFace's model files (one time)

pyclnf uses the model files of [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace)
(Carnegie Mellon University). They are **not included** in pyclnf: OpenFace's license allows
academic or non-profit, non-commercial research use only, and does not allow anyone to pass
the files on. So each user downloads them once, directly from OpenFace's official sources.

1. Open a terminal:
   - **macOS:** open the *Terminal* app.
   - **Windows:** open *Command Prompt* (or *Anaconda Prompt* if you use Anaconda).
   - **Linux:** open a terminal.
2. Type this and press Enter:

   ```bash
   pyclnf-download-models
   ```

   If you see "command not found", type this instead (on Windows you can use `py` in place of `python`):

   ```bash
   python -m pyclnf.download_models
   ```
3. Read the short license summary. If you agree, type `YES` and press Enter.
4. Wait while about 440 MB are downloaded (usually a few minutes). Every file is checked
   automatically. When you see **Done**, pyclnf is ready.

You only do this once per computer and user account. pyclnf then finds the files by itself and
works without an internet connection. If the download is interrupted, run the same command again;
files that were already downloaded and checked are kept.

### Where the files are kept

| System  | Folder |
|---------|--------|
| macOS   | `~/Library/Application Support/OpenFaceModels/2.2.0/` |
| Windows | `%LOCALAPPDATA%\OpenFaceModels\2.2.0\` |
| Linux   | `~/.local/share/OpenFaceModels/2.2.0/` (or `$XDG_DATA_HOME/OpenFaceModels/2.2.0/`) |

The folder is shared with [pyfaceau](https://github.com/johnwilsoniv/pyfaceau) and
[pymtcnn](https://github.com/johnwilsoniv/pymtcnn), so nothing is downloaded twice.
To keep the files somewhere else (for example on another drive), set the environment variable
`OPENFACE_MODELS_DIR` to a folder; pyclnf adds the `2.2.0` sub-folder itself.
To start over, delete the `OpenFaceModels` folder and run `pyclnf-download-models` again.

### Computers without internet

Install the files on a computer that has internet, then copy the whole `OpenFaceModels` folder to
the same place on the offline computer (or set `OPENFACE_MODELS_DIR` to point to it). Keep the
license in mind: the files are for your own non-commercial research and must not be shared with
other people.

### Troubleshooting

- **"certificate verify failed" on macOS:** if you installed Python from python.org, open the Python
  folder in *Applications*, double-click *Install Certificates.command*, then try again.
- **Firewall or proxy:** the files come from `raw.githubusercontent.com` and `github.com`, and the
  four large files from `www.dropbox.com` (`dl.dropboxusercontent.com`), with `onedrive.live.com`
  as OpenFace's fallback.
- **Not enough disk space:** about 550 MB free is needed, or set `OPENFACE_MODELS_DIR` to another drive.

### For developers and apps

```python
from pyclnf import CLNF
from pyclnf.models import ensure_models, ModelsNotInstalledError, LICENSE_URL

# After showing OpenFace's license (LICENSE_URL) in your own dialog:
model_dir = ensure_models(accept_license=True,
                          progress=lambda done_bytes, total_bytes, name: ...)
clnf = CLNF()            # finds the installed files (or: CLNF(model_dir=model_dir))
```

- `CLNF()` never downloads anything by itself. If the files are missing it raises
  `ModelsNotInstalledError` (a `FileNotFoundError`) with plain instructions.
- `ensure_models()` returns immediately (and works offline) once the files are installed.
  Without `accept_license=True` it never downloads; it only prepares files that are already there.
- Setting `OPENFACE_MODELS_ACCEPT_LICENSE=1` counts as accepting the license (for scripts and servers).
- Packaged apps (for example PyInstaller) no longer contain the model files: call `ensure_models`
  on first launch, after your license dialog.
- What happens: the original OpenFace files are downloaded into `originals/`, checked against the
  SHA-256 values in `pyclnf/models/openface_models.json`, and converted into the NumPy files pyclnf
  loads (`derived/pyclnf/1/`). The converted files are byte-for-byte the same as the files that
  pyclnf 0.3.4 shipped, so results are unchanged.

## Usage

```python
from pyclnf import CLNF

clnf = CLNF()
landmarks, info = clnf.detect_and_fit(image)  # 68 facial landmarks
```

For video, call `clnf.detect_and_fit()` (or `clnf.fit()` with a face box) on consecutive frames; it tracks the face across frames.

## What it does

- Detects 68 facial landmarks
- Estimates 3D head pose (pitch, yaw, roll)
- Uses OpenFace's trained CEN patch experts
- Built-in face detection via [pymtcnn](https://github.com/johnwilsoniv/pymtcnn)

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

pyclnf runs OpenFace's models, so please also cite OpenFace (Baltrušaitis et al., *OpenFace 2.0*, IEEE FG 2018).

## License

- **pyclnf's code:** CC BY-NC 4.0, free for non-commercial use with attribution (see `LICENSE`).
- **OpenFace's model files** (downloaded separately, see above): OpenFace's own license,
  academic or non-profit non-commercial research use only, no redistribution or access by third
  parties: <https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt>.
  The files pyclnf converts from them on your computer fall under the same license, which also has
  terms about such derived files; please read it.
