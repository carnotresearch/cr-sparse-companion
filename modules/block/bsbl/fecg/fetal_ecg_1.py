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
import cr.wavelets as crwt
import cr.sparse as crs
import cr.sparse.dict as crdict
import cr.sparse.data as crdata
import cr.sparse.lop as crlop
import cr.sparse.plots as crplot
import cr.sparse.block.bsbl as bsbl
# Sample data
from scipy.misc import electrocardiogram
# Plotting
import matplotlib.pyplot as plt
# Miscellaneous
from scipy.signal import detrend, butter, filtfilt
import scipy.io as sio


signal_01 = sio.loadmat('signal_01.mat')
fs = signal_01['fs']
signals = signal_01['s']

Phi_dict = sio.loadmat('Phi.mat')
Phi = Phi_dict['Phi']
Phi = jnp.asarray(Phi)


sig0 = jnp.asarray(signals[0, :])
X = crn.vec_to_windows(sig0, Phi.shape[1])

# Sensing
Y = Phi @ X
n_windows = X.shape[1]
blk_size = 32
total_time = 0
options = bsbl.bsbl_bo_options(max_iters=20)
for i in range(n_windows):
    x = X[:, i]
    y = Y[:, i]
    start = timeit.default_timer()
    sol = bsbl.bsbl_bo_np_jit(Phi, y, blk_size, options=options)
    stop = timeit.default_timer()
    rtime = stop - start
    total_time += rtime
    x_hat = sol.x
    snr = crn.signal_noise_ratio(x, x_hat)
    prd = crn.percent_rms_diff(x, x_hat)
    nmse = crn.normalized_mse(x, x_hat)
    print(f'[{i}] SNR: {snr:.2f} dB, PRD: {prd:.1f}%, NMSE: {nmse:.5f}, Time: {rtime:.2f} sec, Iters: {sol.iterations}')
print(f'Total reconstruction time: {total_time:.2f} sec')


