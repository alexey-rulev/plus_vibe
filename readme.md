This is a work in progress.

For more details, see: https://doi.org/10.26434/chemrxiv-2025-kf878 or https://doi.org/10.1002/advs.202507261

The app plots projections on band structure; projection DOS is only implemented in ts.ipynb

# Phonon Projection App

The app shows which phonon modes influence the diffusion barrier. In brief, it plots the value of projection of the phonon mode eigenvalue on the vector towards transition state.

The app has GUI using Streamlit.

# Run

In the package directory:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Inputs

- phonon data of a __unit cell__ calculated with phonopy, either:
    - bands.yaml file
    - phonopy.yaml + FORCE_SETS
    - phonopy.yaml + FORCE_CONSTANTS

    For the last to options you should also specify path in Brillouin zone (see defaults for format)

- 3 structure files in ASE-readable format (tested extended xyz, QE .in) with the same number of atoms (ideally all from the same NEB calculation)
    - Initial structure
    - Transition state
    - Final state

Structure files should be supercells constructed from unit cell structure, containig 1 extra diffusing atom (vacancies not yet implemented). The app should automatically guess the replications if they are isotropic (AxAxA); otherwise, input them manually. Index of the extra (diffusing) atom should be specified, default is the last atom (-1).

After all files are uploaded, press "Compute projection" in the bottom left.

## Example

For the reference, see examples folder.

bso_H_diffusion folder contains all files for a proton diffusion in BaSnO<sub>3</sub> with perovskite structure, calculated  with Quantum Espresso. All other inputs can be left on default values; to plot projections in "energy scale" enable "Plot projection^2" checkbox.

currently uploading "bands.yaml" gives the best resuts due to degenerate modes issue.

# To be done

- plottig of projected DOS
- correct plotting of degenerate modes
- vacancies diffusion
- CLI interface
- python API


---
most of the code was generated with LLMs
