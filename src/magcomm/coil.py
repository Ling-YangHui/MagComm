import numpy as np
from magcomm.const import Const
from magcomm.axis_utils import *

class Coil:
    def __init__(self,
                turns: int,                     # 线圈的匝数
                inner_radius: float,            # 线圈的内半径
                radius_width: float,            # 线圈的径宽度
                length: float,                  # 线圈的长度
                has_mag_bar: bool,              # 线圈是否有磁棒
                mag_bar_radius: float,          # 线圈的磁棒半径
                frequency: float,               # 线圈的工作频率
                current: float = 0.0,           # 线圈的工作电流
                towards: str = 'z',             # 线圈的轴朝向（'x', 'y', 'z'）
                phi_axis: str = 'x',            # 极坐标系下的 phi 轴朝向（'x', 'y', 'z'）   
                center_position: np.ndarray = np.zeros(3)   # 线圈的中心位置
                ):
        """
        初始化线圈参数

        Parameters
        ----------
        turns : int
            线圈的匝数
        inner_radius : float
            线圈的内半径
        radius_width : float
            线圈的径宽度
        length : float
            线圈的长度
        has_mag_bar : bool
            线圈是否有磁棒
        mag_bar_radius : float
            线圈的磁棒半径
        frequency : float
            线圈的工作频率
        current : float
            线圈的工作电流
        towards : str
            线圈的轴朝向（'x', 'y', 'z'）
        phi_axis : str
            极坐标系下的 phi 轴朝向（'x', 'y', 'z'）
        """
        # 机械参数
        self.turns = turns
        self.inner_radius = inner_radius
        self.radius_width = radius_width
        self.length = length
        self.has_mag_bar = has_mag_bar
        self.mag_bar_radius = mag_bar_radius

        # 位置参数
        self.towards = towards
        self.center_position = np.asarray(center_position, dtype=float)
        self.phi_axis = phi_axis

        # 电参数
        self.frequency = frequency
        self.current = current

        self.outer_radius = self.inner_radius + self.radius_width                   # 线圈的外半径
        self.mid_radius = np.sqrt(self.inner_radius * self.outer_radius)            # 线圈的中半径
        self.line_length = 2 * np.pi * self.mid_radius * self.turns                 # 线圈的线长
        self.line_area = self.radius_width * self.length / self.turns               # 线圈的线截面积
        self.resistance = Const.pho_copper * self.line_length / self.line_area      # 线圈的电阻

        self.mu_eff_compensate = 1                                                  # 线圈的有效磁导率补偿系数

        if self.has_mag_bar:
            gamma = self.length / (2 * self.mag_bar_radius)                         # 磁芯的长径比
            nd = 1 / (gamma**2 - 1) * (gamma / np.sqrt(gamma**2 - 1) * np.log(gamma + np.sqrt(gamma**2 - 1)) - 1) 
            self.mu_eff = Const.mu_znmn / (1 + nd * (Const.mu_znmn - 1))
            self.area_eff = self.mu_eff * np.pi * self.mag_bar_radius**2 + np.pi * self.mid_radius**2 - np.pi * self.mag_bar_radius**2
        else:
            self.mu_eff = 1.0                                                       # 线圈的有效磁导率
            self.area_eff = np.pi * self.mid_radius**2                              # 线圈的有效面积
        self.mag_moment = self.area_eff * self.turns * self.current                 # 线圈的磁矩 
        self.inductance = Const.mu0 * self.area_eff * self.turns**2 / self.length   # 线圈的电感
        coil_gamma = self.length / (2 * self.mid_radius)                            # 线圈的长径比
        if coil_gamma > 5:
            self.inductance_valid = True
        else:
            self.inductance_valid = False

    def calculate_magnetic_field(self, position: np.ndarray):
        """
        计算线圈在指定位置的磁场强度

        Parameters
        ----------
        position : np.ndarray
            位置坐标，形状为 (3,) 或 (N, 3)，必须为'xyz'坐标系下的笛卡尔坐标

        Returns
        -------
        np.ndarray
            磁场强度，形状为 (N, 3)，xyz坐标系下的笛卡尔坐标
        """
        position, was_1d = to_2d(position)
        self_postion = position_2d(self.center_position, position.shape[0])
        relative_position = position - self_postion
        relative_position_polar = conv_pos_xyz_to_polar(relative_position, theta_axis=self.towards, phi_axis=self.phi_axis)

        b_r = Const.mu0 / (2 * np.pi) * self.mag_moment * np.cos(relative_position_polar[:, 1]) / relative_position_polar[:, 0]**3
        b_theta = Const.mu0 / (4 * np.pi) * self.mag_moment * np.sin(relative_position_polar[:, 1]) / relative_position_polar[:, 0]**3
        b_phi = np.zeros_like(b_r)

        b_polar = np.stack([b_r, b_theta, b_phi], axis=1)
        b_xyz = conv_vec_polar_to_xyz(b_polar, relative_position, position_type='xyz', theta_axis=self.towards, phi_axis=self.phi_axis)
        return b_xyz[0] if was_1d else b_xyz


    def calculate_induced_voltage(self, b_field: np.ndarray, frequency: float = None):
        """
        计算线圈在指定磁场下的感应电压

        Parameters
        ----------
        b_field : np.ndarray
            磁场强度，形状为 (3,) 或 (N, 3) 或 (1,) 或 (N, 1)，其中 (3,) 表示xyz矢量的磁场，(N, 3) 表示 N 个位置的磁场矢量，(1,) 表示标量磁场强度（过轴心），(N, 1) 表示 N 个位置的标量磁场强度（过轴心）
        frequency : float, optional
            频率，默认为线圈的工作频率
        """
        b_field = np.asarray(b_field, dtype=float)
        if frequency is None:
            frequency = self.frequency
        b_field, was_1d = to_2d(b_field)
        if b_field.shape[1] == 3:
            b_field_axis = b_field[:, {'x': 0, 'y': 1, 'z': 2}[self.towards]]
        else:
            b_field_axis = b_field[:, 0]
        induced_voltage = np.abs(2 * np.pi * frequency * self.turns * self.area_eff * b_field_axis)
        return induced_voltage[0] if was_1d else induced_voltage

    def calculate_induced_voltage_to_another_coil(self, other_coil, frequency: float = None):
        """
        计算线圈在另一个线圈的磁场下的感应电压

        Parameters
        ----------
        other_coil : Coil
            另一个线圈对象
        frequency : float, optional
            频率，默认为线圈的工作频率

        Returns
        -------
        np.ndarray
            感应电压
        """
        if frequency is None:
            frequency = self.frequency
        b_field = self.calculate_magnetic_field(other_coil.center_position)
        return other_coil.calculate_induced_voltage(b_field, frequency=frequency)

    def calculate_resonant_voltage(self, inductance: float = None):
        """
        计算线圈的谐振电压

        Returns
        -------
        float
            谐振电压
        """
        if inductance is None:
            inductance = self.inductance
        return np.abs(2 * np.pi * self.frequency * inductance * self.current)

    def calculate_resonant_capacitor(self, inductance: float = None):
        """
        计算线圈的谐振电容

        Returns
        -------
        float
            谐振电容
        """
        if inductance is None:
            inductance = self.inductance
        return 1 / (4 * np.pi**2 * self.frequency**2 * inductance)

class OGCoil(Coil):
    def __init__(self,
                turns=32,
                inner_radius=0.75,
                radius_width=0.1,
                length=0.01, 
                has_mag_bar=False, 
                mag_bar_radius=0.0,
                frequency=4e3,
                current=10,
                towards='z',
                phi_axis='x',
                center_position=np.zeros(3)):
        super().__init__(turns, inner_radius, radius_width, length, has_mag_bar, mag_bar_radius, frequency, current, towards, phi_axis, center_position)

class UGCoil(Coil):
    def __init__(self,
                turns=1200,
                inner_radius=0.018,
                radius_width=0.004,
                length=0.3,
                mag_bar_radius=0.015,
                has_mag_bar=True,
                frequency=3e3,
                current=0.1,
                towards='z',
                phi_axis='x',
                center_position=np.zeros(3)):
        super().__init__(turns, inner_radius, radius_width, length, has_mag_bar, mag_bar_radius, frequency, current, towards, phi_axis, center_position)

class UVCoil(Coil):
    def __init__(self,
                turns = 64,
                inner_radius=0.18,
                radius_width=0.02,
                has_mag_bar=False,
                mag_bar_radius=0.0,
                length=0.01,
                frequency=4e3,
                current=1,
                towards='z',
                phi_axis='x',
                center_position=np.zeros(3)):
        super().__init__(turns, inner_radius, radius_width, length, has_mag_bar, mag_bar_radius, frequency, current, towards, phi_axis, center_position)