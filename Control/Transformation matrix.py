import numpy as np
def transformation_matrix(yaw, pitch, roll):
    
    c_psi = np.cos(yaw)
    s_psi = np.sin(yaw)
    c_theta = np.cos(pitch)
    s_theta = np.sin(pitch)
    c_phi = np.cos(roll)
    s_phi = np.sin(roll)

    C_I_B = np.array([[c_psi * c_theta, c_psi * s_theta * s_phi - s_psi * c_phi, c_psi * s_theta * c_phi + s_psi * s_phi],
                       [s_psi * c_theta, s_psi * s_theta * s_phi + c_psi * c_phi, s_psi * s_theta * c_phi - c_psi * s_phi],
                       [-s_theta, c_theta * s_phi, c_theta * c_phi]])

    
    return C_I_B