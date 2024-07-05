import numpy as np

def dh_transform(a, d, alpha,theta):
    """
    Compute a single transformation matrix using DH parameters.
    """
    # A =  np.array([
    #     [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
    #     [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
    #     [0, np.sin(alpha), np.cos(alpha), d],
    #     [0, 0, 0, 1]
    # ])
    return np.array([[np.cos(theta),-np.sin(theta), 0, a],
                [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                [   0,  0,  0,  1]])

def rotation_matrix_to_euler_angles(R):
    """
    Convert a 3x3 rotation matrix to Euler angles (phi, theta, psi) using Z-Y-X convention.
    
    Args:
    - R (numpy.ndarray): 3x3 rotation matrix

    Returns:
    - tuple: (phi, theta, psi)
    """
    
    # assert R.shape == (3, 3), "Matrix must be of shape 3x3"
    
    # Compute yaw
    psi = np.arctan2(R[1, 0], R[0, 0])
    
    # Compute pitch
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    
    # Compute roll
    phi = np.arctan2(R[2, 1], R[2, 2])
    
    return (phi, theta, psi)

def final_transform(dh_params):
    """
    Compute the overall transformation matrix for the robot manipulator given a list of DH parameters.
    Each entry in the list should be a tuple of (theta, d, a, alpha).
    """
    T_final = np.eye(4)
    for params in dh_params:
        T = dh_transform(params[0],params[1],params[2],params[3])
        T_final = np.dot(T_final, T)
        # T_final = T_final*T
    return T_final

def FK_Panda(q):

    dh_parameters = [
        (0, 0.333, 0, q[0]),
        (0, 0, -np.pi/2, q[1]),
        (0, 0.316, np.pi/2, q[2]),
        (0.0825, 0, np.pi/2, q[3]),
        (-0.0825, 0.384, -np.pi/2, q[4]),
        (0, 0, np.pi/2, q[5]),
        (0.088, 0, np.pi/2, q[6]),
        (0, 0.107, 0, 0)]
    
    Final_Tran_Matrix = final_transform(dh_parameters)
    x_actual = Final_Tran_Matrix[0][3]
    y_actual = Final_Tran_Matrix[1][3]
    z_actual = Final_Tran_Matrix[2][3]
    phi_actual, theta_actual, psi_actual = rotation_matrix_to_euler_angles(Final_Tran_Matrix[:3,:3])

    return [x_actual, y_actual, z_actual, phi_actual, theta_actual, psi_actual]

def compute_jacobian(dh_params):
    """
    Computes the Jacobian matrix for a robot.
    Args:
        dh_params: List of tuples, each containing the DH parameters (a, alpha, d, theta) for each joint.
    Returns:
        Jacobian matrix as a NumPy array.
    """
    num_joints = len(dh_params) - 1              # Number of joints
    jacobian = np.zeros((6, num_joints))
    T = np.identity(4)                            # Initialize to identity matrix
    T_f = T
    for i in range(num_joints+1):
        a, d, alpha, theta = dh_params[i]
        T_i_i = DH_Transform(a, d, alpha, theta)  # Transformation matrix from the previous frame to the i-th frame
        T_f = np.matmul(T_f, T_i_i)               # Overall transformation matrix up to the i-th frame
    p_e = T_f[:3, 3]                              # Extract end-effector position
    T = np.identity(4)                            # Reset transformation matrix
    for i in range(num_joints):
        a, d, alpha, theta = dh_params[i]
        T_i = DH_Transform(a, d, alpha, theta)    # Transformation matrix from the previous frame to the i-th frame
        T = np.matmul(T, T_i)                     # Overall transformation matrix up to the i-th frame
        p = T[:3, 3]                              # Extract the position vector of the i-th frame
        z = T[:3, 2] 
        
        # Calculate linear and angular velocity parts of the Jacobian matrix
        # T = np.matmul(T, T_i)                     # Overall transformation matrix up to the i-th frame
        jacobian[:3, i] = np.cross(z, p_e - p)    # Linear velocity part
        jacobian[3:, i] = z                       # Angular velocity part
    return jacobian

def DH_Transform(a, d, alpha, theta):
    """
    Compute a single transformation matrix using DH parameters.
    Args:
        a: Link length (a)
        d: Link offset (d)
        alpha: Link twist (alpha)
        theta: Joint angle (theta)
    Returns:
        Transformation matrix as a 4x4 NumPy array.
    """
    return np.array([[np.cos(theta),-np.sin(theta), 0, a],
                [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                [   0,  0,  0,  1]])

def DH(q):
    DH = [
    (0, 0.333, 0, q[0]),
    (0, 0, -np.pi/2, q[1]),
    (0, 0.316, np.pi/2, q[2]),
    (0.0825, 0, np.pi/2, q[3]),
    (-0.0825, 0.384, -np.pi/2, q[4]),
    (0, 0, np.pi/2, q[5]),
    (0.088, 0, np.pi/2, q[6]),
    (0, 0.107, 0, 0)]

    return DH


q_prev = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
dh_para = DH(q_prev)
jacobian = compute_jacobian(dh_para)
b= FK_Panda(q_prev)
print(b)
