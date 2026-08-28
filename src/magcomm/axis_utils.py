import numpy as np


_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


def _axis_unit_vector(axis: str) -> np.ndarray:
    """返回坐标轴对应的单位向量（形状 (3,)）。"""
    if axis not in _AXIS_INDEX:
        raise ValueError(f"无效的坐标轴 '{axis}'，可选值为 'x'、'y'、'z'")
    v = np.zeros(3)
    v[_AXIS_INDEX[axis]] = 1.0
    return v


def _polar_basis(theta_axis: str, phi_axis: str):
    """构建极坐标基 (e_a, e_b, e_c)。

    e_a 为极角轴方向，e_b 为方位角参考方向，e_c = e_a × e_b，
    三者构成右手正交基；方位角 phi 由 +e_b 向 +e_c 方向度量。
    """
    if theta_axis not in _AXIS_INDEX:
        raise ValueError(f"无效的 theta_axis '{theta_axis}'，可选值为 'x'、'y'、'z'")
    if phi_axis not in _AXIS_INDEX:
        raise ValueError(f"无效的 phi_axis '{phi_axis}'，可选值为 'x'、'y'、'z'")
    if theta_axis == phi_axis:
        raise ValueError(f"theta_axis 与 phi_axis 不能相同（均为 '{theta_axis}'）")
    e_a = _axis_unit_vector(theta_axis)
    e_b = _axis_unit_vector(phi_axis)
    e_c = np.cross(e_a, e_b)
    return e_a, e_b, e_c


def _xyz_to_angles(pos_xyz: np.ndarray, e_a, e_b, e_c):
    """由 (N, 3) 的笛卡尔位置计算极角 theta 与方位角 phi。"""
    r = np.linalg.norm(pos_xyz, axis=1)
    p_a = pos_xyz @ e_a
    p_b = pos_xyz @ e_b
    p_c = pos_xyz @ e_c
    theta = np.arccos(np.clip(p_a / r, -1.0, 1.0))
    phi = np.arctan2(p_c, p_b)
    return theta, phi

def _get_theta_phi(position, position_type: str, e_a, e_b, e_c):
    """由位置坐标得到极角 theta 与方位角 phi。"""
    if position_type == 'xyz':
        return _xyz_to_angles(position, e_a, e_b, e_c)
    if position_type == 'polar':
        return position[:, 1], position[:, 2]
    raise ValueError(f"无效的 position_type '{position_type}'，可选值为 'xyz' 或 'polar'")

def to_2d(arr):
    """
    将输入数组规整为 (N, 3) 的二维数组，支持 (3,) 广播与 (N, 3)。

    Parameters
    ----------
    arr : np.ndarray
        输入数组，形状为 (3,) 或 (N, 3)
    
    Returns
    -------
    np.ndarray
        规整后的二维数组，形状为 (N, 3)
    bool
        如果输入为 (3,) 则返回 True，否则返回 False

    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, False


def position_2d(position, n: int):
    """
    将位置坐标规整为 (N, 3) 的二维数组，支持 (3,) 广播与 (N, 3)。

    Parameters
    ----------
    position : np.ndarray
        位置坐标，形状为 (3,) 或 (N, 3)
    n : int
        期望的行数 N

    Returns
    -------
    np.ndarray
        规整后的二维数组，形状为 (N, 3)
    """
    pos = np.asarray(position, dtype=float)
    if pos.ndim == 1:
        return np.broadcast_to(pos, (n, 3))
    if pos.ndim == 2 and pos.shape[1] == 3:
        if pos.shape[0] != n:
            raise ValueError(f"position 行数 {pos.shape[0]} 与向量行数 {n} 不一致")
        return pos
    

def conv_pos_xyz_to_polar(
    position: np.ndarray,
    theta_axis: str = 'z',
    phi_axis: str = 'x'
):
    """
    将三维笛卡尔坐标系下的位置坐标转换为极坐标系下的位置坐标

    Parameters
    ----------
    position : np.ndarray
        三维笛卡尔坐标系下的位置坐标，形状为 (3,) 或 (N, 3)
    theta_axis : str, optional
        极坐标系下的极角轴，默认为 'z'
    phi_axis : str, optional
        极坐标系下的方位角轴，默认为 'x'

    Returns
    -------
    np.ndarray
        极坐标系下的位置坐标 (r, theta, phi)，形状与输入一致
    """
    e_a, e_b, e_c = _polar_basis(theta_axis, phi_axis)
    pos2d, was_1d = to_2d(position)
    theta, phi = _xyz_to_angles(pos2d, e_a, e_b, e_c)
    r = np.linalg.norm(pos2d, axis=1)
    out = np.stack([r, theta, phi], axis=1)
    return out[0] if was_1d else out


def conv_pos_polar_to_xyz(
    position: np.ndarray,
    theta_axis: str = 'z',
    phi_axis: str = 'x'
):
    """
    将极坐标系下的位置坐标转换为三维笛卡尔坐标系下的位置坐标

    Parameters
    ----------
    position : np.ndarray
        极坐标系下的位置坐标 (r, theta, phi)，形状为 (3,) 或 (N, 3)
    theta_axis : str, optional
        极坐标系下的极角轴，默认为 'z'
    phi_axis : str, optional
        极坐标系下的方位角轴，默认为 'x'

    Returns
    -------
    np.ndarray
        三维笛卡尔坐标系下的位置坐标，形状与输入一致
    """
    e_a, e_b, e_c = _polar_basis(theta_axis, phi_axis)
    pos2d, was_1d = to_2d(position)
    r = pos2d[:, 0]
    theta = pos2d[:, 1]
    phi = pos2d[:, 2]
    p_a = r * np.cos(theta)
    p_b = r * np.sin(theta) * np.cos(phi)
    p_c = r * np.sin(theta) * np.sin(phi)
    out = p_a[:, None] * e_a + p_b[:, None] * e_b + p_c[:, None] * e_c
    return out[0] if was_1d else out


def conv_vec_xyz_to_polar(
    v_xyz: np.ndarray,
    position: np.ndarray,
    position_type: str = 'xyz',
    theta_axis: str = 'z',
    phi_axis: str = 'x'
):
    """
    将三维笛卡尔坐标系下的向量转换为极坐标系下的向量

    Parameters
    ----------
    v_xyz : np.ndarray
        三维笛卡尔坐标系下的向量，形状为 (3,) 或 (N, 3)
    position : np.ndarray
        位置坐标，形状为 (3,) 或 (N, 3)
    position_type : str, optional
        位置坐标的类型，默认为 'xyz'，表示笛卡尔坐标系，可选值为 'xyz' 或 'polar'
    theta_axis : str, optional
        极坐标系下的极角轴，默认为 'z'
    phi_axis : str, optional
        极坐标系下的方位角轴，默认为 'x'

    Returns
    -------
    np.ndarray
        极坐标系下的向量 (v_r, v_theta, v_phi)，形状与 v_xyz 一致
    """
    e_a, e_b, e_c = _polar_basis(theta_axis, phi_axis)
    v2d, was_1d = to_2d(v_xyz)
    pos2d = position_2d(position, v2d.shape[0])
    theta, phi = _get_theta_phi(pos2d, position_type, e_a, e_b, e_c)

    # 向量在 (e_b, e_c, e_a) 基下的分量
    v_b = v2d @ e_b
    v_c = v2d @ e_c
    v_a = v2d @ e_a

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    v_r = st * cp * v_b + st * sp * v_c + ct * v_a
    v_theta = ct * cp * v_b + ct * sp * v_c - st * v_a
    v_phi = -sp * v_b + cp * v_c

    out = np.stack([v_r, v_theta, v_phi], axis=1)
    return out[0] if was_1d else out


def conv_vec_polar_to_xyz(
    v_polar: np.ndarray,
    position: np.ndarray,
    position_type: str = 'xyz',
    theta_axis: str = 'z',
    phi_axis: str = 'x'
):
    """
    将极坐标系下的向量转换为三维笛卡尔坐标系下的向量

    Parameters
    ----------
    v_polar : np.ndarray
        极坐标系下的向量 (v_r, v_theta, v_phi)，形状为 (3,) 或 (N, 3)
    position : np.ndarray
        位置坐标，形状为 (3,) 或 (N, 3)
    position_type : str, optional
        位置坐标的类型，默认为 'xyz'，表示笛卡尔坐标系，可选值为 'xyz' 或 'polar'
    theta_axis : str, optional
        极坐标系下的极角轴，默认为 'z'
    phi_axis : str, optional
        极坐标系下的方位角轴，默认为 'x'

    Returns
    -------
    np.ndarray
        三维笛卡尔坐标系下的向量，形状与 v_polar 一致
    """
    e_a, e_b, e_c = _polar_basis(theta_axis, phi_axis)
    v2d, was_1d = to_2d(v_polar)
    pos2d = position_2d(position, v2d.shape[0])
    theta, phi = _get_theta_phi(pos2d, position_type, e_a, e_b, e_c)

    v_r = v2d[:, 0]
    v_theta = v2d[:, 1]
    v_phi = v2d[:, 2]

    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    v_b = st * cp * v_r + ct * cp * v_theta - sp * v_phi
    v_c = st * sp * v_r + ct * sp * v_theta + cp * v_phi
    v_a = ct * v_r - st * v_theta

    out = v_a[:, None] * e_a + v_b[:, None] * e_b + v_c[:, None] * e_c
    return out[0] if was_1d else out
