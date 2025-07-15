from scipy.signal import butter, filtfilt, detrend
import numpy as np

def nf(data):
    "Null Filter. Does nothing."
    return data

def butter_lowpass_filter(data, cutoff=10, fs=30.0, order=4):
    """
    Apply a low-pass Butterworth filter to multivariate time series data.

    Parameters:
        data: np.ndarray of shape (time, sensors)
        cutoff: cutoff frequency (Hz)
        fs: sampling frequency (Hz)
        order: filter order

    Returns:
        filtered_data: np.ndarray of shape (time, sensors)
    """
    data = np.array(data)

    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq

    [b, a] = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = np.zeros_like(data)

    # Apply the filter to each sensor (column) independently
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])

    return filtered_data

def butter_detrend(data, cutoff=10, fs=30.0, order=4, poly_order=4):
    """
    Apply a low-pass Butterworth filter to multivariate time series data, with
    signal detrending.

    Parameters:
        data: np.ndarray of shape (time, sensors)
        cutoff: cutoff frequency (Hz)
        fs: sampling frequency (Hz)
        order: filter order
        poly_order: polynomial order for de-trending

    Returns:
        filtered_data: np.ndarray of shape (time, sensors)
    """
    data = np.array(data)
    n_samples, n_channels = data.shape
    t = np.arange(n_samples)

    # Detrend each column by removing a polynomial trend
    detrended_data = np.zeros_like(data)
    for i in range(n_channels):
        coeffs = np.polyfit(t, data[:, i], poly_order)
        trend = np.polyval(coeffs, t)
        detrended_data[:, i] = data[:, i] - trend

    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq

    [b, a] = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = np.zeros_like(data)

    # Apply the filter to each sensor (column) independently
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, detrended_data[:, i])

    return filtered_data
