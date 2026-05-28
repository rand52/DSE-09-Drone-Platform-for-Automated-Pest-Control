import numpy as np
from scipy.spatial.transform import Rotation as R

#Coordinate frame transformations:
# Uses numpy so should be given in radians
def euler_to_quat(phi, theta, psi):
    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_psi = np.cos(psi)
    s_psi = np.sin(psi)
    R = [[c_theta*c_psi, c_theta*s_psi, -s_theta],
         [(s_phi*s_theta*c_psi)-(c_phi*s_psi), (s_phi*s_theta*s_psi)+(c_phi*c_psi), s_phi*c_theta],
         [(c_phi*s_theta*c_psi)+(s_phi*s_psi), (c_phi*s_theta*s_psi)-(s_phi*c_psi), c_phi*c_theta]]
    r = R.from_matrix(R)
    print(R.as_quat())

    return R.as_quat()

R = euler_to_quat(1, 1, 1)

