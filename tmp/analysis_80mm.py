from magcomm.coil import *
from magcomm.const import Const
from magcomm.axis_utils import *

y = 300

tx_coil = OGCoil(
    towards='z',
    phi_axis='x'
)

print(f"接收位置磁场强度：{tx_coil.calculate_magnetic_field(np.array([0, y, -30])) / Const.mu0} A/m")

rx_coil = UGCoil(
    turns=1200,
    inner_radius=0.005,
    radius_width=0.001,
    length=0.081,
    mag_bar_radius=0.004,
    has_mag_bar=True,
    center_position=np.array([0, y, -30]),
    current=0.05,
    # real_line=True,
    # line_radius=0.0005,
    towards='z',
    phi_axis='x'
)

uv_coil = UVCoil(
    turns=200,
    center_position=np.array([0, y, 0]),
    inner_radius=0.1,
    radius_width=0.002,
    length=0.03
)

print(f"接收线圈电感: {rx_coil.inductance:.6e} H")
print(f"接收线圈电阻: {rx_coil.resistance:.6e} Ohm")


print(f"下行开环感应电压：{tx_coil.calculate_induced_voltage_to_another_coil(rx_coil, frequency=4e3):.6e} V")
print(f"上行开环感应电压：{rx_coil.calculate_induced_voltage_to_another_coil(uv_coil):.6e} V")