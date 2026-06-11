import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform

#Coordinate frame transformations:
# Uses numpy so should be given in radians
def euler_to_quat(psi, theta, phi):
    # c_phi = np.cos(phi)
    # s_phi = np.sin(phi)
    # c_theta = np.cos(theta)
    # s_theta = np.sin(theta)
    # c_psi = np.cos(psi)
    # s_psi = np.sin(psi)
    # H = [[c_theta*c_psi, c_theta*s_psi, -s_theta],
    #      [(s_phi*s_theta*c_psi)-(c_phi*s_psi), (s_phi*s_theta*s_psi)+(c_phi*c_psi), s_phi*c_theta],
    #      [(c_phi*s_theta*c_psi)+(s_phi*s_psi), (c_phi*s_theta*s_psi)-(s_phi*c_psi), c_phi*c_theta]]
    # h = R.from_matrix(H)
    i = R.from_euler('zyx', [psi, theta, phi], degrees=False)
    x, y, z, w = i.as_quat()
    return np.array([w, x, y, z])
# angular rate should be a vector containing [roll rate, pitch rate, yaw rate] q should be the quaternion
def rates_euler_to_quat(angular_rate, q):
    # cross_vec = [0]
    # cross_vec.append(angular_rate)
    print(([0] + angular_rate))
    q_dot = 0.5 * quat_multiply(q, [0]+ angular_rate)
    return q_dot

def quat_integrate(q, q_dot, dt):
    q = q + q_dot * dt
    q = q / np.linalg.norm(q)  # Normalize the quaternion to maintain unit length   
    return q


def quat_multiply(p, q):
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    t0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    t1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    t2 = p0*q2 - p1*q3 + p2*q0 - p3*q1
    t3 = p0*q3 + p1*q2 - p2*q1 - p3*q0
    return np.array([t0, t1, t2, t3])

j = euler_to_quat(np.pi/2, 0, -np.pi/2)
print('after switch', j)
rates= rates_euler_to_quat([2.0, 3.0, 3.0], j)
print(rates)

