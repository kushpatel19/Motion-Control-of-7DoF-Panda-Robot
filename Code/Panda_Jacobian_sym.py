import numpy as np
import sympy as sp

def dh_transform(a, d, alpha, theta):
    return np.array([
        [sp.cos(theta), -sp.sin(theta) * sp.cos(alpha), sp.sin(theta) * sp.sin(alpha), a * sp.cos(theta)],
        [sp.sin(theta), sp.cos(theta) * sp.cos(alpha), -sp.cos(theta) * sp.sin(alpha), a * sp.sin(theta)],
        [0, sp.sin(alpha), sp.cos(alpha), d],
        [0, 0, 0, 1]
    ], dtype=object)

def compute_jacobian(dh_params):
    num_joints = len(dh_params) - 1
    jacobian = np.zeros((6, num_joints), dtype=object)
    T = np.eye(4, dtype=object)
    
    for i in range(num_joints + 1):
        a, d, alpha, theta = dh_params[i]
        T_i_i = dh_transform(a, d, alpha, theta)
        T = np.dot(T, T_i_i)
    
    p_e = T[:3, 3]
    T = np.eye(4, dtype=object)
    
    for i in range(num_joints):
        a, d, alpha, theta = dh_params[i]
        T_i = dh_transform(a, d, alpha, theta)
        T = np.dot(T, T_i)
        p = T[:3, 3]
        z = T[:3, 2]
        
        jacobian[:3, i] = np.cross(z, p_e - p)
        jacobian[3:, i] = z
    
    return jacobian

# Define symbolic joint angles
q1, q2, q3, q4, q5, q6, q7 = sp.symbols('q1 q2 q3 q4 q5 q6 q7')

# Create symbolic DH parameters
sym_dh_parameters = [
    (0, 0.333, 0, q1),
    (0, 0, -sp.pi/2, q2),
    (0, 0.316, sp.pi/2, q3),
    (0.0825, 0, sp.pi/2, q4),
    (-0.0825, 0.384, -sp.pi/2, q5),
    (0, 0, sp.pi/2, q6),
    (0.088, 0, sp.pi/2, q7),
    (0, 0.107, 0, 0)
]

# Compute symbolic Jacobian
sym_jacobian = compute_jacobian(sym_dh_parameters)
print(sym_jacobian)
