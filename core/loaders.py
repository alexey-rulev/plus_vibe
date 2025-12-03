import yaml

from .phonon_parser import parse_band_yaml


def load_band_yaml_from_bytes(b: bytes):
    data = yaml.safe_load(b.decode('utf-8', errors='ignore'))
    return parse_band_yaml(data)
