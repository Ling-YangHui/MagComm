import numpy as np

class Const:
    mu0 = 4 * np.pi * 1e-7          # 真空磁导率
    c = 3e8                         # 光速
    pho_copper = 1.68e-8            # 铜的电阻率
    sigma_copper = 1 / pho_copper   # 铜的电导率
    epsilon0 = 8.854187817e-12      # 真空介电常数
    mu_znmn = 2000                  # 锌锰合金的相对磁导率
    pho_zmmn = 1                    # 锌锰合金的电阻率
    sigma_znmn = 1 / pho_zmmn       # 锌锰合金的电导率 