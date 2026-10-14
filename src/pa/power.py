import numpy as np
from scipy.integrate import trapezoid

def calculate_power(voltage: np.ndarray, current: np.ndarray, time: np.ndarray) -> float:
    """
    计算在一段时间内的平均功率

    Parameters
    ----------
    voltage : np.ndarray
        电压随时间变化的数组
    current : np.ndarray
        电流随时间变化的数组
    time : np.ndarray
        时间数组，单位为秒

    Returns
    -------
    float
        平均功率，单位为瓦特
    """
    instantaneous_power = voltage * current
    average_power = trapezoid(instantaneous_power, time) / (time[-1] - time[0])
    return average_power