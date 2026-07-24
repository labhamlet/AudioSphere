import scipy
import scipy.special
print(f"SciPy version: {scipy.__version__}")
try:
    from scipy.special import sph_harm
    print("sph_harm found in scipy.special")
except ImportError:
    print("sph_harm NOT found in scipy.special")
    # Check if it is available under another name or submodule
    if hasattr(scipy.special, 'sph_harm'):
        print("scipy.special.sph_harm exists")
    else:
        print("scipy.special.sph_harm does NOT exist")
