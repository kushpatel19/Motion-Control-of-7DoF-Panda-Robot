import numpy as np

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

def cartesian_velocity(jacobian, joint_velocities):
    """
    Computes the Cartesian velocities (linear and angular) of the end-effector.

    jacobian: numpy array (6xn), Jacobian matrix.
    joint_velocities: numpy array (nx1), joint space velocities.

    Returns:
    cartesian_vel: numpy array (6x1), Cartesian velocities. 
                   The first three elements are linear velocities, 
                   and the last three are angular velocities.
    """
    
    # Calculate the Cartesian velocities
    cartesian_vel = np.matmul(jacobian, joint_velocities)
    
    return cartesian_vel

def joint_velocity(jacobian, cart_vel):
    """
    Computes joint velocities from Cartesian velocities using the Jacobian matrix.
    Args:
        jacobian: Jacobian matrix (6xN)
        cart_vel: Cartesian velocities (6x1)
    Returns:
        Joint velocities as a NumPy array (Nx1).
    """
    return np.matmul(np.linalg.pinv(jacobian),cart_vel)

def Manipulability_ellipsoid(jacobian):
    """
    Computes the singular values of the Jacobian for the Manipulability Ellipsoid.
    Args:
        jacobian: Jacobian matrix (6xN)
    Returns:
        Array of singular values.
    """
    U, S, Vt = np.linalg.svd(jacobian)
    return np.array(S)

def force_ellipsoid(jacobian):
    """
    Computes the singular values of the Jacobian for the Force Ellipsoid.
    Args:
        jacobian: Jacobian matrix (6xN)
    Returns:
        Array of singular values reciprocals.
    """
    U, S, Vt = np.linalg.svd(jacobian)
    return 1/np.array(S)

def manipulability_index(jacobian):
    """
    Computes the manipulability index.
    Args:
        jacobian: Jacobian matrix (6xN)
    Returns:
        Manipulability index value.
    """
    return np.sqrt(np.abs(np.linalg.det(np.matmul(jacobian,np.transpose(jacobian)))))

# def generate_dh_parameters(link_lengths, joint_angles):
#     """
#     Generates DH parameters for an n-DOF planar robot
#     link_lengths: List containing the lengths of the links
#     joint_angles: List containing the joint angles
#     Returns: List of tuples, each containing the DH parameters (a, alpha, d) for each joint
#     """
#     if len(link_lengths) != len(joint_angles):
#         raise ValueError("Link lengths and joint angles lists must have the same length")
    
#     dh_parameters = []
#     for i in range(len(link_lengths)):
#         a = link_lengths[i]
#         theta = joint_angles[i]
#         alpha = 0      # alpha is 0 for planar robots
#         d = 0          # d is 0 for revolute joints in planar robots
#         dh_parameters.append((a, alpha, d, theta))
#     return dh_parameters


# # Example Usage:
# n = 3  # Number of DOFs of each finger 
# link_lengths = [2, 2, 2]  # Replace with the actual link lengths of your robot
# joint_angles = [np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]  # Replace with the actual joint angles of your robot
# dh_params = generate_dh_parameters(link_lengths, joint_angles)

q = [-1, -2.5, 0.5, 1.5, -2, -0.01, 0.56]
dh_params = np.array([
        (0, 0.333, 0, q[0]),
        (0, 0, -np.pi/2, q[1]),
        (0, 0.316, np.pi/2, q[2]),
        (0.0825, 0, np.pi/2, q[3]),
        (-0.0825, 0.384, -np.pi/2, q[4]),
        (0, 0, np.pi/2, q[5]),
        (0.088, 0, np.pi/2, q[6]),
        (0, 0.107, 0, 0)
        # (0, 0.1034, 0, np.pi/4)
    ])

# You can then use dh_params directly with the previously provided compute_jacobian function
# jacobian_matrix = np.round(compute_jacobian(dh_params),6)
# jacobian_matrix = compute_jacobian(dh_params)
# print(np.round(jacobian_matrix,3))

# joint_vel = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 0]

# cart_vel = cartesian_velocity(jacobian_matrix, joint_vel)
# print()
# print(cart_vel)

# w = manipulability_index(jacobian_matrix)
# print()
# print(w)

# print(np.round(np.matmul(np.transpose(jacobian_matrix),jacobian_matrix),3))