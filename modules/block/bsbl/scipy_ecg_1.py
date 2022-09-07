# Configure JAX for 64-bit computing
from jax.config import config
config.update("jax_enable_x64", True)

import timeit
import jax
import numpy as np
import jax.numpy as jnp

# CR-Suite libraries
import cr.nimble as crn
import cr.nimble.dsp as crdsp
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.block.bsbl as bsbl

# Sample data
from scipy.misc import electrocardiogram
# Miscellaneous
from scipy.signal import detrend, butter, filtfilt

def main():

    ecg = electrocardiogram()
    # Sampling frequency in Hz
    fs = 360

    N = 1000
    max_iters = 20

    # We shall only process one signal in this demo
    x = ecg[:N]
    x = detrend(x)
    ## bandpass filter
    # lower cutoff frequency
    f1 = 5
    # upper cutoff frequency
    f2 = 40
    # passband in normalized frequency
    Wn = np.array([f1, f2]) * 2 / fs
    # butterworth filter
    fn = 3
    fb, fa = butter(fn, Wn, 'bandpass')
    x = filtfilt(fb,fa,x)
    CR = 0.5
    M = int(N * CR)
    print(f'M={M}, N={N}, CR={CR}')
    Phi = crdict.gaussian_mtx(crn.KEY0, M, N)
    y = Phi @ x

    # BSBL-EM
    start = timeit.default_timer()
    options = bsbl.bsbl_em_options(y, prune_gamma=0, max_iters=max_iters)
    sol = bsbl.bsbl_em_jit(Phi, y, 25, options=options)
    stop = timeit.default_timer()
    print(f'Time: {stop - start:.2f} sec', )
    print(sol)
    x_hat = sol.x
    print(f'BSBL-EM: SNR: {crn.signal_noise_ratio(x, x_hat):.2f} dB, PRD: {crn.percent_rms_diff(x, x_hat):.1f}%')

    # BSBL-BO
    options = bsbl.bsbl_bo_options(y, prune_gamma=0, max_iters=max_iters)
    start = timeit.default_timer()
    sol = bsbl.bsbl_bo_jit(Phi, y, 25, options=options)
    stop = timeit.default_timer()
    print(f'Time: {stop - start:.2f} sec', )
    print(sol)
    x_hat = sol.x
    print(f'BSBL-BO: SNR: {crn.signal_noise_ratio(x, x_hat):.2f} dB, PRD: {crn.percent_rms_diff(x, x_hat):.1f}%')

    # BSBL-BO-NP
    options = bsbl.bsbl_bo_options(y, max_iters=max_iters)
    start = timeit.default_timer()
    sol = bsbl.bsbl_bo_np_jit(Phi, y, 25, options=options)
    stop = timeit.default_timer()
    print(f'Time: {stop - start:.2f} sec', )
    print(sol)
    x_hat = sol.x
    print(f'BSBL-BO-NP: SNR: {crn.signal_noise_ratio(x, x_hat):.2f} dB, PRD: {crn.percent_rms_diff(x, x_hat):.1f}%')

if __name__ == '__main__':
    main()
