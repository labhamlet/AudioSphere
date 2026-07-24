import numpy as np
from scipy.special import factorial as fact

def wigner_d_matrix(N: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute the Wigner-D matrix for Spherical Harmonics rotation.

    The matrix D rotates SH coefficients such that:
    f_rot(omega) = f(R^-1 omega)
    c_rot = D(R) @ c

    Parameters
    ----------
    N : int
        Maximum SH order.
    alpha, beta, gamma : float
        Euler angles in radians (Z-Y-Z convention).
        Rotation R = Rz(alpha) * Ry(beta) * Rz(gamma).

    Returns
    -------
    D : np.ndarray
        Wigner-D matrix of shape ((N+1)^2, (N+1)^2).
        Block diagonal structure with blocks of size (2n+1)x(2n+1).
    """
    # Total number of coefficients
    L = (N + 1) ** 2
    D = np.zeros((L, L), dtype=np.complex128)

    # Compute for each order n
    for n in range(N + 1):
        # Get the small-d matrix for this order
        d_n = _wigner_small_d(n, beta)

        # Construct the full D matrix for this order
        # D^n_{m',m} = e^{-i m' alpha} * d^n_{m',m}(beta) * e^{-i m gamma}

        m_range = np.arange(-n, n + 1)

        # Phase terms
        # exp(-i * m' * alpha)  [rows]
        phase_left = np.exp(-1j * m_range * alpha)

        # exp(-i * m * gamma)   [cols]
        phase_right = np.exp(-1j * m_range * gamma)

        # Combine: D = diag(phase_left) @ d @ diag(phase_right)
        # Broadcasting: (2n+1, 1) * (2n+1, 2n+1) * (1, 2n+1)
        D_n = phase_left[:, np.newaxis] * d_n * phase_right[np.newaxis, :]

        # Place in the big matrix
        start_idx = n**2
        end_idx = (n + 1) ** 2
        D[start_idx:end_idx, start_idx:end_idx] = D_n

    return D


def _wigner_small_d(j: int, beta: float) -> np.ndarray:
    """
    Compute the Wigner small-d matrix d^j(beta) for a specific order j.
    Uses recursion or explicit formula. For stability and speed, we use
    scipy's rotation logic or a robust recursion.

    Here we implement a robust recursion for d^j_{m',m}.
    """
    # Size of the matrix for order j is (2j+1) x (2j+1)
    dim = 2 * j + 1
    d = np.zeros((dim, dim))

    # Indices map from 0..2j to -j..j
    # m = idx - j

    # Base case j=0
    if j == 0:
        return np.array([[1.0]])

    # For j=1/2 (spinor) we know the formula, but we need integer j.
    # We can use the relation to Jacobi polynomials or recursion.

    # Using the formula involving Jacobi polynomials P^(a,b)_n(x)
    # d^j_{m',m}(beta) = ...
    # This is numerically stable.
    # However, scipy doesn't have a direct "wigner d" function exposed easily.

    # Let's use a simple recursion on j?
    # Or calculate via rotation of vector SH?

    # Actually, for integer j, we can use the explicit formula:
    # d^j_{m',m}(\beta) = \sum_k (-1)^k ...
    # This is slow and unstable for large j.

    # Better: Use the fact that d^j is related to rotation of the basis.
    # But implementing it from scratch is error prone.

    # Let's use the 'spherical' library approach or a simplified recursion.
    # A very stable method is evaluating the d-matrix using FFT of rotations? No.

    # Let's use the standard recursion on j (Trapani & Navaza 2006).
    # But that requires implementing it.

    # ALTERNATIVE:
    # Use scipy.spatial.transform.Rotation to rotate a set of points,
    # then project back? That's slow.

    # Let's implement the explicit formula for low orders (N<=20 is fine).
    # Or use `scipy.special.jacobi`.

    # d^j_{m',m}(beta) = xi * sqrt(...) * (cos(b/2))^a * (sin(b/2))^b * P_k^(a,b)(cos beta)
    # where k = j - m', a = m' - m, b = m' + m (if m'>=m)
    # We need to handle symmetries.

    from scipy.special import eval_jacobi

    cb = np.cos(beta)
    sb_2 = np.sin(beta / 2.0)
    cb_2 = np.cos(beta / 2.0)

    for m_prime_idx in range(dim):
        m_prime = m_prime_idx - j
        for m_idx in range(dim):
            m = m_idx - j

            # Use symmetry to ensure a, b >= 0 for Jacobi
            # d^j_{m',m} = (-1)^(m'-m) d^j_{m,m'}
            # d^j_{m',m} = (-1)^(m'-m) d^j_{-m',-m}

            # We compute for the case where we can map to Jacobi P_k^(a,b)
            # Formula from Wikipedia / Edmonds:
            # k = j - m_prime
            # But we need a, b >= 0.

            # Let's use the generic formula with factorials (stable for N<20-30)
            # Or just use the Jacobi one carefully.

            # Case 1: m' >= m
            # Let mu = |m' - m|
            # Let nu = |m' + m|
            # s = j - (mu + nu)/2
            # This is getting complicated.

            # Let's use the `pyshtools` logic simplified.
            # d(beta) is real.

            val = _calc_d_element(j, m_prime, m, beta)
            d[m_prime_idx, m_idx] = val

    return d


def _calc_d_element(j, mp, m, beta):
    """Calculate single element d^j_{mp, m}(beta)."""
    # Using the formula related to Jacobi polynomials
    # d^j_{m',m}(b) = C * (sin(b/2))^(m-m') * (cos(b/2))^(m+m') * P_{j-m}^{(m-m', m+m')}(cos b)
    # This is valid for m >= m'.

    # Symmetries to map to valid Jacobi indices (alpha, beta > -1)
    # d^j_{m',m} = (-1)^(j-m) d^j_{m', -m} (No)

    # Symmetry: d^j_{m',m} = (-1)^(m'-m) d^j_{m,m'}
    # Symmetry: d^j_{m',m} = d^j_{-m,-m'}

    # We map to case where k >= 0 and alpha, beta >= 0

    # We use the expression:
    # d^j_{m',m} = sqrt( (j+m)! (j-m)! / ((j+m')! (j-m')!) ) * (sin b/2)^(m'-m) * (cos b/2)^(m'+m) * P_{j-m'}^{(m'-m, m'+m)}(cos b)
    # Valid for m' >= m?

    # Let's just implement the sum formula which is robust for small j.
    # d^j_{m',m}(beta) = sum_k (-1)^k * ...
    # This is stable enough for N=20.

    # sqrt((j+m')!(j-m')!(j+m)!(j-m)!) * sum_k [ (-1)^k / (k! (j-m'-k)! (j+m-k)! (m'-m+k)!) ] * (cos b/2)^(2j - m' + m - 2k) * (sin b/2)^(m' - m + 2k)

    # Precompute trig
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    factor = np.sqrt(fact(j + mp) * fact(j - mp) * fact(j + m) * fact(j - m))

    res = 0.0

    # Range of k:
    # k >= 0
    # j - mp - k >= 0  => k <= j - mp
    # j + m - k >= 0   => k <= j + m
    # mp - m + k >= 0  => k >= m - mp

    k_min = max(0, m - mp)
    k_max = min(j - mp, j + m)

    for k in range(k_min, k_max + 1):
        denom = fact(k) * fact(j - mp - k) * fact(j + m - k) * fact(mp - m + k)
        term = (
            (-1) ** k * (c ** (2 * j - mp + m - 2 * k)) * (s ** (mp - m + 2 * k))
        ) / denom
        res += term

    return factor * res
