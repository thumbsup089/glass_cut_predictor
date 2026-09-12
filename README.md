# Glass Cut Predictor

Creates radial-heightmap and sparse mesh-flattening, geometry-only blank predictions.
STL coordinates are preserved, units must be millimetres, and `(0, 0)` is the radial origin.

## Install and run (Windows / Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy the mold to `data/stl/mold.stl` (or edit `STL_PATH`), then run `python main.py`.
SVGs go to `output/svg`; debug meshes and plots go to `output/debug`.

## Geometric limitation

Doubly-curved surfaces cannot generally be flattened without stretching, compression,
or cuts. Both outputs are geometry-only baselines, not physical glass-slumping simulations.
