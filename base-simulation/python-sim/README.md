# Python Base Simulation

This folder contains the first attempt of the thesis base simulation.

There are two scripts:

- `src/personal-simulation/personal_simulation.py`
- `src/official-simulation/official_simulation.py`

## Install

From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run

- Run personal pendulum simulation:

```bash
python src/personal-simulation/personal_simulation.py
```

- Run official Gymnasium demo:

```bash
python src/official-simulation/official_simulation.py
```

Both scripts open a local matplotlib window and show an animated view.
