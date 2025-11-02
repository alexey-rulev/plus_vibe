This is a work in progress.

Currently the working code is in ts.ipynb (test branch)

# Phonon Projection Streamlit App

Loads three `band.yaml` files (unit-cell with eigenvectors, supercell equilibrium, supercell TS),
computes supercell displacements, folds to the unit cell with q-dependent phases, and projects onto unit-cell modes.
Marker size in the band diagram is proportional to the projection amplitude.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
