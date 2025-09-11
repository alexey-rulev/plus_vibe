"""Utilities for loading phonopy band.yaml files."""

import io
from phonopy import load as phonopy_load


def load_band_yaml(file_obj):
    """Return ``(phonon, band)`` from a ``band.yaml`` file.

    Parameters
    ----------
    file_obj : str or file-like
        Path to ``band.yaml`` or an object with ``read()`` returning the file
        contents.

    Returns
    -------
    phonon : :class:`phonopy.Phonopy`
        Reconstructed Phonopy instance.
    band : :class:`phonopy.phonon.band_structure.BandStructure`
        Band structure data associated with ``phonon``.
    """
    if hasattr(file_obj, "read"):
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode()
        file_obj = io.StringIO(content)
    band = phonopy_load(band_yaml=file_obj)
    return band.phonon, band
