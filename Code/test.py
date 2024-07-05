import sympy as sp
import numpy as np

def symbolic_DH_Transform(a, d, alpha, theta):
    """
    Compute a single transformation matrix using DH parameters symbolically.
    Args:
        a: Link length (a)
        d: Link offset (d)
        alpha: Link twist (alpha)
        theta: Joint angle (theta)
    Returns:
        Transformation matrix as a 4x4 SymPy matrix.
    """
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ])

def symbolic_compute_jacobian(dh_params):
    """
    Computes the symbolic Jacobian matrix for a robot.
    Args:
        dh_params: List of tuples, each containing the symbolic DH parameters (a, alpha, d, theta) for each joint.
    Returns:
        Jacobian matrix as a 6xN SymPy matrix.
    """
    num_joints = len(dh_params) - 1  # Number of joints
    jacobian = sp.zeros(6, num_joints)
    T = sp.eye(4)  # Initialize to identity matrix
    T_f = T
    for i in range(num_joints + 1):
        a, d, alpha, theta = dh_params[i]
        T_i_i = symbolic_DH_Transform(a, d, alpha, theta)  # Transformation matrix from the previous frame to the i-th frame
        T_f = T_f * T_i_i  # Overall transformation matrix up to the i-th frame
    p_e = T_f[:3, 3]  # Extract end-effector position
    T = sp.eye(4)  # Reset transformation matrix
    for i in range(num_joints):
        a, d, alpha, theta = dh_params[i]
        T_i = symbolic_DH_Transform(a, d, alpha, theta)  # Transformation matrix from the previous frame to the i-th frame
        T = T * T_i  # Overall transformation matrix up to the i-th frame
        p = T[:3, 3]  # Extract the position vector of the i-th frame
        z = T[:3, 2]
        jacobian[:3, i] = z.cross(p_e - p)  # Linear velocity part
        jacobian[3:, i] = z  # Angular velocity part
    return jacobian

# Define symbolic variables for joint angles
theta1, theta2, theta3, theta4, theta5, theta6, theta7 = sp.symbols('theta1:8')
# Define a fixed symbolic 6x1 force vector
force_vector = sp.Matrix([sp.symbols('F_{x}'), sp.symbols('F_{y}'), sp.symbols('F_{z}'),
                          sp.symbols('M_{x}'), sp.symbols('M_{y}'), sp.symbols('M_{z}')])


# Create symbolic DH parameters for your robot
symbolic_dh_params = [
    (0, 0.333, 0, theta1),
    (0, 0, -sp.pi/2, theta2),
    (0, 0.316, sp.pi/2, theta3),
    (0.0825, 0, sp.pi/2, theta4),
    (-0.0825, 0.384, -sp.pi/2, theta5),
    (0, 0, sp.pi/2, theta6),
    (0.088, 0, sp.pi/2, theta7),
    (0, 0.107, 0, 0)
    # (0, 0.1034, 0, sp.pi/4)
]

# Compute the symbolic Jacobian
symbolic_jacobian = symbolic_compute_jacobian(symbolic_dh_params)

# Differentiate the transpose of the Jacobian matrix with respect to each joint angle
diff_matrices = []  # To store the derivative matrices
for i in range(len(symbolic_dh_params) - 1):
    diff_matrix = sp.diff(symbolic_jacobian.T, symbolic_dh_params[i][-1])
    # print((diff_matrix))
    diff_matrices.append(diff_matrix)

# Multiply each derivative matrix with the fixed force vector and create a 7x7 matrix
matrix_list = []  # To store 7x1 matrices after multiplication
# Define a 7x7 NumPy array filled with zeros
original_array = np.zeros((7, 7), dtype=object)

for i, diff_matrix in enumerate(diff_matrices):
    matrix_list.append(diff_matrix * force_vector)
    column_array = np.array(diff_matrix * force_vector).astype(object)
    original_array[:,i] = column_array[:,0]

# print(np.array(symbolic_jacobian).shape)
# print(np.array(diff_matrices).shape)

# Print the symbolic Jacobian
# for i in range(symbolic_jacobian.shape[0]):
#     for j in range(symbolic_jacobian.shape[1]):
#         print(f'J_{i+1}{j+1} =', symbolic_jacobian[i, j])

# # Print the derivative matrices
# for i, diff_matrix in enumerate(diff_matrices):
#     print(f'Derivative of J.T w.r.t theta{i + 1}:')
#     sp.pprint(diff_matrix)
#     print()

# Create a 7x7 matrix where each column is a 7x1 matrix
final_matrix = sp.Matrix(matrix_list).T
# print(np.array(final_matrix).shape)
# print(original_array)
# Print the 7x7 matrix
# sp.pprint(final_matrix)

# Define numerical values for the symbolic variables
numerical_values = {
    theta1: -1,  # Replace with actual values
    theta2: -2.5,
    theta3: 0.5,
    theta4: 1.5,
    theta5: -2,
    theta6: -0.01,
    theta7: 0.56,
    sp.symbols('F_{x}'): 1.0,  # Replace with actual force values
    sp.symbols('F_{y}'): 2.0,
    sp.symbols('F_{z}'): 3.0,
    sp.symbols('M_{x}'): 0.1,
    sp.symbols('M_{y}'): 0.2,
    sp.symbols('M_{z}'): 0.3
}

# Replace symbolic values with numerical values in the final matrix
final_matrix_numeric = sp.Matrix(original_array).subs(numerical_values)
final_value_numeric = np.array(final_matrix_numeric,dtype=float)
print(final_value_numeric)