# Thesis Preparation Repo

This repo is for the first thesis base simulations.

Folder used now:

- `base-simulation/matlab/`
  - `personal-simulation.m` (simple custom inverse pendulum simulation)
  - `official-simulation-example.m` (opens the official MathWorks example)

- `base-simulation/python-sim/`
  - `src/personal-simulation/personal_simulation.py` (simple custom simulation, animated)
  - `src/official-simulation/official_simulation.py` (Gymnasium based simulation, animated)
  - `pyproject.toml` and `README.md` for install/run steps

How to run Python:

```bash
cd base-simulation/python-sim
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python src/personal-simulation/personal_simulation.py
python src/official-simulation/official_simulation.py
```

This is a work-in-progress base version only.
