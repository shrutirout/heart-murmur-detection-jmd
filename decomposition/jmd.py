"""Jump Plus AM-FM Mode Decomposition (JMD) - Memory Efficient Version"""

import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve


def mirror_extend_matlab(signal):
    """Mirror-extend signal to length 2T."""
    T = len(signal)
    T_half = T // 2
    f = np.zeros(2 * T)
    f[0:T_half] = signal[T_half-1::-1]
    f[T_half:T_half + T] = signal
    f[T_half + T:2*T] = signal[T-1:T_half-1:-1]
    return f


def compute_alpha_schedule(alpha, N=2000):
    """Compute alpha warm-up schedule."""
    a2 = 50
    t2 = np.arange(0.01, np.sqrt(2/a2) + 0.001, 0.001)
    phi1 = (-a2/2) * (t2**2) + (np.sqrt(2*a2) * t2)
    phi = np.concatenate([phi1, np.ones(N - len(phi1))])
    return alpha * phi


def JMD(signal, alpha=5000, tau=5, beta=0.03, b_bar=0.45, K=3, init=0, tol=1e-6):
    """
    Jump Plus AM-FM Mode Decomposition (JMD) - Memory Efficient.

    Parameters:
        signal : 1D array
        alpha  : bandwidth constraint
        tau    : dual ascent step
        beta   : jump constraint (1/expected jumps)
        b_bar  : jump height parameter
        K      : number of modes
        init   : 0=zeros, 1=uniform, 2=random
        tol    : convergence tolerance

    Returns:
        u     : (K, T) decomposed modes
        v     : (T,) jump component
        omega : (K,) final center frequencies
    """
    shift = np.mean(signal)
    signal = signal - shift
    save_T = len(signal)
    fs = 1.0 / save_T

    f = mirror_extend_matlab(signal)
    T = len(f)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1/T

    N = 2000
    Alpha = compute_alpha_schedule(alpha, N)

    f_hat = fftshift(fft(f))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T//2] = 0

    u_hat_curr = np.zeros((T, K), dtype=np.complex128)
    u_hat_prev = np.zeros((T, K), dtype=np.complex128)
    omega_curr = np.zeros(K)
    omega_prev = np.zeros(K)

    if init == 1:
        for i in range(K):
            omega_prev[i] = (0.5 / K) * i
    elif init == 2:
        omega_prev[:] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))

    uDiff = tol + np.finfo(float).eps
    n = 0

    b = 2.0 / (b_bar ** 2)
    gamma = tau * (0.5 * b * beta)
    v = np.zeros(T)

    d = np.ones(T)
    D = spdiags(np.vstack([-d, d]), np.array([0, 1]), T, T, format="lil")
    D[-1, :] = 0
    D = D.tocsr()
    DTD = D.T @ D

    x = np.zeros(T)
    rho = np.zeros(T)
    coef1 = 1.0 / gamma if gamma > 0 else 0
    mu = 2 * beta / gamma if gamma > 0 else 0
    SPDiag = speye(T, format="csr")
    j_hat_curr = np.zeros(T, dtype=np.complex128)
    j_hat_prev = np.zeros(T, dtype=np.complex128)

    if gamma > 0:
        v_update_matrix = SPDiag + gamma * DTD

    while uDiff > tol and n < N - 1:
        sum_uk = np.sum(u_hat_prev, axis=1) - u_hat_prev[:, 0]

        for k in range(K):
            if k > 0:
                sum_uk = sum_uk + u_hat_curr[:, k-1] - u_hat_prev[:, k]

            denom = 1 + Alpha[n] * (freqs - omega_prev[k])**2
            u_hat_curr[:, k] = (f_hat_plus - sum_uk - j_hat_prev) / denom

            pos_freqs = freqs[T//2:]
            pos_power = np.abs(u_hat_curr[T//2:, k])**2
            sum_power = np.sum(pos_power)
            if sum_power > 0:
                omega_curr[k] = np.sum(pos_freqs * pos_power) / sum_power

        u_hat_full = np.zeros((T, K), dtype=np.complex128)
        for k in range(K):
            u_hat_full[T//2:, k] = u_hat_curr[T//2:, k]
            u_hat_full[T//2:0:-1, k] = np.conj(u_hat_curr[T//2:, k])
            u_hat_full[0, k] = np.conj(u_hat_full[-1, k])

        u = np.zeros((K, T))
        for k in range(K):
            u[k, :] = np.real(ifft(ifftshift(u_hat_full[:, k])))

        if gamma > 0:
            rhs = (gamma * D.T @ x - D.T @ rho) + f - np.sum(u, axis=0)
            v = spsolve(v_update_matrix, rhs)
        else:
            v = f - np.sum(u, axis=0)

        Dv = D @ v
        h = Dv + coef1 * rho

        if mu * b < 1:
            abs_h = np.abs(h) + 1e-10
            scale = np.clip((1.0 / (1 - mu * b)) - (mu * np.sqrt(2 * b) / (1 - mu * b)) / abs_h, 0, 1)
            x = scale * h
        else:
            x = np.zeros_like(h)

        rho = rho - gamma * (x - Dv)
        v = v - (np.mean(v) - np.mean(f))

        j_hat_curr[:] = fftshift(fft(v))
        j_hat_curr[:T//2] = 0

        n += 1

        uDiff = np.finfo(float).eps
        for k in range(K):
            diff = u_hat_curr[:, k] - u_hat_prev[:, k]
            uDiff += (1/T) * np.sum(diff * np.conj(diff)).real
        j_diff = j_hat_curr - j_hat_prev
        uDiff += (1/T) * np.sum(j_diff * np.conj(j_diff)).real
        uDiff = np.abs(uDiff)

        u_hat_prev[:] = u_hat_curr
        omega_prev[:] = omega_curr
        j_hat_prev[:] = j_hat_curr

    u_hat_final = np.zeros((T, K), dtype=np.complex128)
    u_hat_final[T//2:, :] = u_hat_curr[T//2:, :]
    u_hat_final[T//2:0:-1, :] = np.conj(u_hat_curr[T//2:, :])
    u_hat_final[0, :] = np.conj(u_hat_final[-1, :])

    u = np.zeros((K, T))
    for k in range(K):
        u[k, :] = np.real(ifft(ifftshift(u_hat_final[:, k])))

    u = u[:, T//4:3*T//4]
    v = v[T//4:3*T//4] + shift

    sort_idx = np.argsort(omega_curr)
    u = u[sort_idx, :]
    omega_final = omega_curr[sort_idx]

    return u, v, omega_final
