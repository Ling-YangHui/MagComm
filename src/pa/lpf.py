import numpy as np
import scipy.signal as signal

def fir_lowpass_filter(data, cutoff_freq, fs, order=5):
    """
    Apply a FIR low-pass filter to the input data.

    Parameters:
    data (array-like): The input signal to be filtered.
    cutoff_freq (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling frequency of the input signal in Hz.
    order (int): The order of the filter. Default is 5.

    Returns:
    array-like: The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    # Design the FIR filter using the window method
    taps = signal.firwin(order + 1, normal_cutoff)
    # Apply the filter to the data
    filtered_data = signal.lfilter(taps, 1.0, data)
    return filtered_data

def iir_lowpass_filter(data, cutoff_freq, fs, order=5):
    """
    Apply an IIR low-pass filter to the input data.

    Parameters:
    data (array-like): The input signal to be filtered.
    cutoff_freq (float): The cutoff frequency of the filter in Hz.
    fs (float): The sampling frequency of the input signal in Hz.
    order (int): The order of the filter. Default is 5.

    Returns:
    array-like: The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    # Design the IIR filter using a Butterworth design
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter to the data
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data