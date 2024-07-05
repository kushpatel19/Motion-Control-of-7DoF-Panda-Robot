import numpy as np
import math
from Panda_IK import franka_IK_EE
from Panda_Jacobian import compute_jacobian, manipulability_index, cartesian_velocity, joint_velocity, Manipulability_ellipsoid, force_ellipsoid
import matplotlib.pyplot as plt

def python_2dlist_to_lua_table(py_2dlist):
    """
    Converts a 2D Python list to a Lua table represented as a string.
    
    Args:
    - py_2dlist (list of lists): 2D Python list to be converted.

    Returns:
    - str: String representation of a Lua table.
    """
    def list_to_lua_row(py_list):
        return "{" + ", ".join(map(str, py_list)) + "}"
    
    lua_rows = [list_to_lua_row(row) for row in py_2dlist]
    return "{\n" + ",\n".join(lua_rows) + "\n}"

def joint_trajectory(t, q_min, q_max):
    midpoint = (q_min + q_max) / 2
    amplitude = (q_max - q_min) / 2
    
    # Set the angular frequency
    omega = 2 * np.pi / (t[-1] - t[0])  # One full sine wave over the time range
    
    # Generate a random phase shift for variety
    phi = np.random.uniform(0, 2*np.pi, size=q_min.shape)

    # Reshape t for broadcasting
    t_reshaped = t.reshape(-1, 1)
    
    # trajectory = np.outer(np.ones(t.shape), midpoint) + np.outer(np.sin(omega * t + phi), amplitude)
    # trajectory = np.outer(np.ones(t.shape), midpoint) + np.outer(np.sin(omega * t_reshaped + phi), amplitude)
    trajectory = midpoint + np.sin(omega * t_reshaped + phi) * amplitude
    vel_trajectory = np.cos(omega*t_reshaped + phi)*amplitude*omega

    return trajectory, vel_trajectory

def first_valid_row(arr):
    for row in arr:
        if not np.any(np.isnan(row)):
            return row
    return "AK47"  # Return None if no valid row is found

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

# def joint_space_velocity(t, trajectory):
#     dt = t[1] - t[0]  # Assuming uniform time steps
#     velocity = np.zeros_like(trajectory)

#     # For the inner points, use central difference
#     velocity[1:-1] = (trajectory[2:] - trajectory[:-2]) / (2*dt)

#     # For the boundaries, use forward and backward difference
#     velocity[0] = (trajectory[1] - trajectory[0]) / dt
#     velocity[-1] = (trajectory[-1] - trajectory[-2]) / dt

#     return velocity

# def random_joints(N):
#     q_min = np.array([-2.8972, -1.7627, -2.8972, -3.0717, -2.8972, -0.0174, -2.8972])
#     q_max = np.array([2.8972, 1.7627, 2.8972, -0.0697, 2.8972, 3.7524, 2.8972])
    
#     # Generate N random 7-dimensional vectors within the given range
#     random_vectors = np.random.uniform(q_min, q_max, (N, 7))
    
#     return random_vectors

# N = 10000
# joint_set = random_joints(N)
# # print(ve)

t = np.linspace(0, 10, 500)  # Time range from 0 to 10 with 500 steps
q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

trajectory, joint_vel_trajectory = joint_trajectory(t,q_min,q_max)
trajectory = np.array([row + [0] for row in (np.array(trajectory)[:,:6]).tolist()])
j_vel = np.array([row + [0] for row in (np.array(joint_vel_trajectory)[:,:6]).tolist()])
# j_vel = np.array(list(joint_space_velocity(t,trajectory)[:6]).append(0))

# Example usage:
count = 0

q_prev = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
Q_pred = []
X_actual = []
Y_actual = []
Z_actual = []
Phi_actual = []
Theta_actual = []
Psi_actual = []

X_pred = []
Y_pred = []
Z_pred = []
Phi_pred = []
Theta_pred = []
Psi_pred = []

W = []
Cart_vel_actual = []
Cart_vel_pred = []
Joint_vel_pred = []

Sigma_Man = []
Sigma_Force = []

for i in range(len(trajectory)):
    q = trajectory[i]

    dh_parameters = [
        (0, 0.333, 0, q[0]),
        (0, 0, -np.pi/2, q[1]),
        (0, 0.316, np.pi/2, q[2]),
        (0.0825, 0, np.pi/2, q[3]),
        (-0.0825, 0.384, -np.pi/2, q[4]),
        (0, 0, np.pi/2, q[5]),
        (0.088, 0, np.pi/2, q[6]),
        (0, 0.107, 0, 0)
        # (0, 0.1034, 0, np.pi/4)
    ]
    Final_Tran_Mat = final_transform(dh_parameters)

    x_actual = Final_Tran_Mat[0][3]
    y_actual = Final_Tran_Mat[1][3]
    z_actual = Final_Tran_Mat[2][3]
    phi_actual, theta_actual, psi_actual = rotation_matrix_to_euler_angles(Final_Tran_Mat[:3,:3])

    X_actual.append(x_actual)
    Y_actual.append(y_actual)
    Z_actual.append(z_actual)
    Phi_actual.append(phi_actual)
    Theta_actual.append(theta_actual)
    Psi_actual.append(psi_actual)

    O_T_EE_array = Final_Tran_Mat
    q7 = q[6]
    q_actual_array = q_prev

    results = franka_IK_EE(O_T_EE_array, q7, q_actual_array)

    # q_pred = results[0]
    q_pred = first_valid_row(results)
    if q_pred == "AK47":
        q_pred = q_prev
    Q_pred.append(q_pred)
    q_prev = q_pred
    dh_parameters_pred = [
        (0, 0.333, 0, q_pred[0]),
        (0, 0, -np.pi/2, q_pred[1]),
        (0, 0.316, np.pi/2, q_pred[2]),
        (0.0825, 0, np.pi/2, q_pred[3]),
        (-0.0825, 0.384, -np.pi/2, q_pred[4]),
        (0, 0, np.pi/2, q_pred[5]),
        (0.088, 0, np.pi/2, q_pred[6]),
        (0, 0.107, 0, 0)
        # (0, 0.1034, 0, np.pi/4)
    ]
    Final_Tran_Mat_pred = final_transform(dh_parameters_pred)

    x_pred = Final_Tran_Mat_pred[0][3]
    y_pred = Final_Tran_Mat_pred[1][3]
    z_pred = Final_Tran_Mat_pred[2][3]
    phi_pred, theta_pred, psi_pred = rotation_matrix_to_euler_angles(Final_Tran_Mat_pred[:3,:3])

    X_pred.append(x_pred)
    Y_pred.append(y_pred)
    Z_pred.append(z_pred)
    Phi_pred.append(phi_pred)
    Theta_pred.append(theta_pred)
    Psi_pred.append(psi_pred)

    if np.array_equal(np.round(Final_Tran_Mat,6), np.round(Final_Tran_Mat_pred,6)):
        count+=1
    else:
        pass
    
    jacob_actual = compute_jacobian(dh_parameters)
    jacob_pred = compute_jacobian(dh_parameters_pred)
    w = manipulability_index(jacob_pred)
    W.append(w)

    cart_vel_actual = cartesian_velocity(jacob_actual, j_vel[i])
    j_vel_pred = np.array(list(joint_velocity(jacob_pred[:6,:6],cart_vel_actual)) + [0])  #setting q7_dot = 0 (redundancy parameter)

    # j_vel_pred = np.array([row + [0] for row in (np.array(joint_velocity(jacob_pred[:6,:6],cart_vel_actual))[:6]).tolist()]) #setting q7_dot = 0 (redundancy parameter)
    cart_vel_pred = cartesian_velocity(jacob_pred,j_vel_pred)

    Joint_vel_pred.append(j_vel_pred)
    Cart_vel_actual.append(cart_vel_actual)
    Cart_vel_pred.append(cart_vel_pred)

    s_m = Manipulability_ellipsoid(jacob_pred)
    Sigma_Man.append(s_m)

    s_f = force_ellipsoid(jacob_pred)
    Sigma_Force.append(s_f)    

Joint_vel_pred = np.array(Joint_vel_pred)
Cart_vel_actual = np.array(Cart_vel_actual)
Cart_vel_pred = np.array(Cart_vel_pred)
Sigma_Man = np.array(Sigma_Man)
Sigma_Force = np.array(Sigma_Force)

print(count)

# Q_pred = np.array(Q_pred)
# print(Q_pred)
# print(python_2dlist_to_lua_table(Q_pred))
# print(X_actual)
# Plot the results

# Plotting the trajectories for visualization
for i in range(7):
    plt.plot(t, trajectory[:, i], label=f'Joint {i+1}')
plt.xlabel('Time')
plt.ylabel('Joint Value')
plt.legend()
plt.title('Time-dependent Joint Trajectories')
plt.grid(True)
plt.show()

### Plot comparsion of actual vs predicted caartesian trajectory
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(t, X_actual, label='Actual x')
plt.plot(t, X_pred, label='Predicted x', linestyle='--')
plt.title('x vs. Time')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t, Y_actual, label='Actual y')
plt.plot(t, Y_pred, label='Predicted y', linestyle='--')
plt.title('y vs. Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t, Z_actual, label='Actual z')
plt.plot(t, Z_pred, label='Predicted z', linestyle='--')
plt.title('z vs. Time')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(t, Phi_actual, label='Actual phi')
plt.plot(t, Phi_pred, label='Predicted phi', linestyle='--')
plt.title('phi vs. Time')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t, Theta_actual, label='Actual theta')
plt.plot(t, Theta_pred, label='Predicted theta', linestyle='--')
plt.title('theta vs. Time')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t, Psi_actual, label='Actual psi')
plt.plot(t, Psi_pred, label='Predicted psi', linestyle='--')
plt.title('psi vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

### Plot comparsion of actual vs predicted caartesian velocities
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(t, Cart_vel_actual[:,0], label='Actual xdot')
plt.plot(t, Cart_vel_pred[:,0], label='Predicted xdot', linestyle='--')
plt.title('xdot vs. Time')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t, Cart_vel_actual[:,1], label='Actual ydot')
plt.plot(t, Cart_vel_pred[:,1], label='Predicted ydot', linestyle='--')
plt.title('ydot vs. Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t, Cart_vel_actual[:,2], label='Actual zdot')
plt.plot(t, Cart_vel_pred[:,2], label='Predicted zdot', linestyle='--')
plt.title('zdot vs. Time')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(t, Cart_vel_actual[:,3], label='Actual phi_dot')
plt.plot(t, Cart_vel_pred[:,3], label='Predicted phi_dot', linestyle='--')
plt.title('phi_dot vs. Time')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t, Cart_vel_actual[:,4], label='Actual theta_dot')
plt.plot(t, Cart_vel_pred[:,4], label='Predicted theta_dot', linestyle='--')
plt.title('theta_dot vs. Time')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t, Cart_vel_actual[:,5], label='Actual psi_dot')
plt.plot(t, Cart_vel_pred[:,5], label='Predicted psi_dot', linestyle='--')
plt.title('psi_dot vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

### Plot joint velocities vs time and manipulability index
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(t, Joint_vel_pred[:,0], label='Predicted Joint1 Velocity')
plt.title('q1dot vs. Time')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(t, Joint_vel_pred[:,1], label='Predicted Joint2 Velocity')
plt.title('q2dot vs. Time')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(t, Joint_vel_pred[:,2], label='Predicted Joint3 Velocity')
plt.title('q3dot vs. Time')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(t, Joint_vel_pred[:,3], label='Predicted Joint4 Velocity')
plt.title('q4dot vs. Time')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(t, Joint_vel_pred[:,4], label='Predicted Joint5 Velocity')
plt.title('q5dot vs. Time')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(t, Joint_vel_pred[:,5], label='Predicted Joint6 Velocity')
plt.title('q6dot vs. Time')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(t, Joint_vel_pred[:,6], label='Predicted Joint7 Velocity')
plt.title('q7dot vs. Time')
plt.legend()

plt.subplot(2, 4, 8)
plt.plot(t, W, label='Manipulability Index')
plt.title('w vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

### Plot Singular Values of Manipulablity Ellipsoid vs time
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(t, Sigma_Man[:,0], label='1st Singular value of Manipulablity Ellipsoid')
plt.title('Sigma1 vs. Time')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t, Sigma_Man[:,1], label='2nd Singular value of Manipulablity Ellipsoid')
plt.title('Sigma2 vs. Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t, Sigma_Man[:,2], label='3rd Singular value of Manipulablity Ellipsoid')
plt.title('Sigma3 vs. Time')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(t, Sigma_Man[:,3], label='4th Singular value of Manipulablity Ellipsoid')
plt.title('Sigma4 vs. Time')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t, Sigma_Man[:,4], label='5th Singular value of Manipulablity Ellipsoid')
plt.title('Sigma5 vs. Time')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t, Sigma_Man[:,5], label='6th Singular value of Manipulablity Ellipsoid')
plt.title('Sigma6 vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

### Plot Singular Values of Force Ellipsoid vs time
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(t, Sigma_Force[:,0], label='1st Singular value of Force Ellipsoid')
plt.title('Sigma1 vs. Time')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t, Sigma_Force[:,1], label='2nd Singular value of Force Ellipsoid')
plt.title('Sigma2 vs. Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t, Sigma_Force[:,2], label='3rd Singular value of Force Ellipsoid')
plt.title('Sigma3 vs. Time')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(t, Sigma_Force[:,3], label='4th Singular value of Force Ellipsoid')
plt.title('Sigma4 vs. Time')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t, Sigma_Force[:,4], label='5th Singular value of Force Ellipsoid')
plt.title('Sigma5 vs. Time')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t, Sigma_Force[:,5], label='6th Singular value of Force Ellipsoid')
plt.title('Sigma6 vs. Time')
plt.legend()

plt.tight_layout()
plt.show()
