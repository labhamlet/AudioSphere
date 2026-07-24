import os

# Base directory of the package (src/shroom)
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")

# Paths
DEFAULT_HRTF_PATH = os.path.join(DATA_DIR, "default_hrtf.sofa")
DEFAULT_WAV_PATH = os.path.join(DATA_DIR, "speech.wav")
