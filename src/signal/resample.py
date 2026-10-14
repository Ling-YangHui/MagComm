import numpy as np
import scipy.signal as signal
from fractions import Fraction

def resample_signal(data, timestamp, new_fs):
    """
    Resample the input signal to a new sampling frequency.

    Parameters:
    data (array-like): The input signal to be resampled.
    timestamp (array-like): The original timestamps of the input signal, it can be uniformly spaced.
    new_fs (float): The new sampling frequency in Hz.

    Returns:
    tuple: A tuple containing the resampled signal and the new timestamps.
    """
    data = np.asarray(data)
    timestamp = np.asarray(timestamp, dtype=float)

    # Estimate the original sampling frequency from the median sample
    # interval so that occasional jitter or outliers do not bias it.
    dt = np.diff(timestamp)
    old_fs = 1.0 / np.median(dt)

    # If the samples are not uniformly spaced, first interpolate them onto a
    # uniform grid so that a band-limited resampler can be applied.
    if not np.allclose(dt, dt[0]):
        n_uniform = int(np.ceil((timestamp[-1] - timestamp[0]) * old_fs)) + 1
        t_uniform = timestamp[0] + np.arange(n_uniform) / old_fs
        if np.iscomplexobj(data):
            data = (np.interp(t_uniform, timestamp, data.real)
                    + 1j * np.interp(t_uniform, timestamp, data.imag))
        else:
            data = np.interp(t_uniform, timestamp, data)
        timestamp = t_uniform

    # Resample using a rational ratio between the old and new sample rates.
    ratio = Fraction(new_fs / old_fs).limit_denominator()
    resampled_data = signal.resample_poly(data, ratio.numerator, ratio.denominator)

    # Build the corresponding new timestamps.
    new_timestamp = timestamp[0] + np.arange(len(resampled_data)) / new_fs

    return resampled_data, new_timestamp