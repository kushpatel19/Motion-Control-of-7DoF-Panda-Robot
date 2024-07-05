import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
import matlab.engine

# Start the MATLAB engine
eng = matlab.engine.start_matlab()

# Add the folder containing your .m file to the MATLAB path (if it's not already there)
eng.addpath('get_MassMatrix.m')
eng.addpath('get_CoriolisVector.m')
eng.addpath('get_FrictionTorque.m')
eng.addpath('get_GravityVector.m')

# Define the robot parameters
l1 = 2
l2 = 2
l3 = 2

m1 = 1
m2 = 1
m3 = 1

g = 9.8

#Forward Kinematics
def DH_Parameters(q):
    dh_parameters = np.array([
    (0, 0.333, 0, q[0]),
    (0, 0, -np.pi/2, q[1]),
    (0, 0.316, np.pi/2, q[2]),
    (0.0825, 0, np.pi/2, q[3]),
    (-0.0825, 0.384, -np.pi/2, q[4]),
    (0, 0, np.pi/2, q[5]),
    (0.088, 0, np.pi/2, q[6]),
    (0, 0.107, 0, 0)])
        
    return dh_parameters    

def dh_transform(a, d, alpha,theta):
    return np.array([[np.cos(theta),-np.sin(theta), 0, a],
                [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                [   0,  0,  0,  1]])

def final_transform(dh_params):
    T_final = np.eye(4)
    for params in dh_params:
        T = dh_transform(params[0],params[1],params[2],params[3])
        T_final = np.dot(T_final, T)
        # T_final = T_final*T
    return T_final

def rotation_matrix_to_euler_angles(R):    
    # assert R.shape == (3, 3), "Matrix must be of shape 3x3"
    # Compute yaw
    psi = np.arctan2(R[1, 0], R[0, 0])
    # Compute pitch
    theta = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    # Compute roll
    phi = np.arctan2(R[2, 1], R[2, 2])
    return (phi, theta, psi)

# Forwared Kinematics
def forward_kin(dh_params):
    T = final_transform(dh_params)
    R = T[:3,:3]
    phi,theta,psi = rotation_matrix_to_euler_angles(R)
    x,y,z = T[0,3], T[1,3], T[2,3]
    return np.array([x,y,z,phi,theta,psi])

# Forward Velocity Kinematics
def forward_vel_kin(J,qdot):
    xdot = np.matmul(J,qdot)
    return xdot

# Jacobian Computation
def Jacob(dh_params):
    """
    Computes the Jacobian matrix
    dh_params: List of tuples, each tuple contains the DH parameters (a, alpha, d) for each joint.
    q: List of joint angles.
    """
    num_joints = len(dh_params) - 1
    jacobian = np.zeros((6, num_joints))
    T = np.identity(4)                            # Initialize to identity matrix
    T_f = T
    for i in range(num_joints+1):
        a, d, alpha, theta = dh_params[i]
        T_i_i = dh_transform(a, d, alpha, theta)  # Transformation matrix from the previous frame to the i-th frame
        T_f = np.matmul(T_f, T_i_i)               # Overall transformation matrix up to the i-th frame
    p_e = T_f[:3, 3]
    # print(p_e)
    T = np.identity(4)
    for i in range(num_joints):
        a, d, alpha, theta = dh_params[i]
        T_i = dh_transform(a, d, alpha, theta)    # Transformation matrix from the previous frame to the i-th frame
        T = np.matmul(T, T_i)                     # Overall transformation matrix up to the i-th frame
        p = T[:3, 3]                              # Extract the position vector of the i-th frame
        z = T[:3, 2] 
        # print(z)
        # print(p) 
        # print(np.cross(z, p_e - p))   
        # print()                         # Extract the z-axis of the i-th frame (third column of T)
        # T = np.matmul(T, T_i)                     # Overall transformation matrix up to the i-th frame
        jacobian[:3, i] = np.cross(z, p_e - p)    # Linear velocity part
        jacobian[3:, i] = z                       # Angular velocity part
    return jacobian

# def Jacob_dot(q,qdot):
#     theta1 = q[0]
#     theta2 = q[1]
#     theta3 = q[2]

#     theta1dot = qdot[0]
#     theta2dot = qdot[1]
#     theta3dot = qdot[2]

#     Jdot = np.array([
#         [-l1*(theta1dot)*cos(theta1) - l2*(theta1dot+theta2dot)*cos(theta1 + theta2) - l3*(theta1dot+theta2dot+theta3dot)*cos(theta1 + theta2 + theta3),
#         -l2*(theta1dot+theta2dot)*cos(theta1 + theta2) - l3*(theta1dot+theta2dot+theta3dot)*cos(theta1 + theta2 + theta3),
#         -l3*(theta1dot+theta2dot+theta3dot)*cos(theta1 + theta2 + theta3)],
#         [-(l1*(theta1dot)*sin(theta1) + l2*(theta1dot+theta2dot)*sin(theta1 + theta2) + l3*(theta1dot+theta2dot+theta3dot)*sin(theta1 + theta2 + theta3)),
#         -(l2*(theta1dot+theta2dot)*sin(theta1 + theta2) + l3*(theta1dot+theta2dot+theta3dot)*sin(theta1 + theta2 + theta3)),
#         -(l3*(theta1dot+theta2dot+theta3dot)*sin(theta1 + theta2 + theta3))]
#     ])  
#     return Jdot

# Numerical Inverse Kinematics
def ik_solver(q_prev, qdot_prev, dt):
    q_next = q_prev + qdot_prev*dt
    return q_next

# Resolving Redundancy Using Optimization Criterion
def red_resolver(q,xdot,type,qdot_prev = None,step_size = None):
    # if type == "fixed_orientation":
    #     J = Jacob(q,"3dim")
    #     xdot = np.array([xdot[0],xdot[1],np.sum(qdot_prev)])
    #     qdot = np.matmul(np.linalg.inv(J),xdot)
    # dh_params = np.array([
    #         (0, 0.333, 0, q[0]),
    #         (0, 0, -np.pi/2, q[1]),
    #         (0, 0.316, np.pi/2, q[2]),
    #         (0.0825, 0, np.pi/2, q[3]),
    #         (-0.0825, 0.384, -np.pi/2, q[4]),
    #         (0, 0, np.pi/2, q[5]),
    #         (0.088, 0, np.pi/2, q[6]),
    #         (0, 0.107, 0, 0)])
    dh_params = DH_Parameters(q)
    if type == "min_norm":
        J = Jacob(dh_params)
        qdot = np.matmul(np.linalg.pinv(J),xdot)

    elif type == "min_joint_vel":
        J = Jacob(dh_params)
        for i in range(1):
            qdot = np.matmul(np.linalg.pinv(J),xdot) - step_size*np.matmul((np.identity(7) - np.matmul(np.linalg.pinv(J),J)), qdot_prev)
            # qdot = np.matmul(np.linalg.pinv(J),xdot) - step_size*np.matmul((np.identity(3) - np.matmul(np.linalg.pinv(J),J)), np.array([1,1,1]))
            # qdot = np.matmul(np.linalg.pinv(J),xdot) - step_size*np.matmul((np.identity(3) - np.matmul(np.linalg.pinv(J),J)), 3*qdot_prev**2)
            qdot_prev = qdot
    
    return qdot

# Computing the Joint Trajectory
def joint_traj(q_initial,qdot_initial,Xdot,type,dt, step_size = None):
    q_traj = [q_initial]
    qdot_traj = [qdot_initial]
    qdot_prev = qdot_initial
    q_prev = q_initial
    for i in range(len(Xdot)-1):
        # dt = (x_traj[i+1] - x_traj[i])/xdot_traj[i]
        q_next = ik_solver(q_prev,qdot_prev,dt)
        if type == "min_joint_vel":
            q_dot_next = red_resolver(q_next,np.array(Xdot[i+1]),type,qdot_prev,step_size=step_size)
        else:
            q_dot_next = red_resolver(q_next,np.array(Xdot[i+1]),type,qdot_prev)
        q_traj.append(q_next)
        qdot_traj.append(q_dot_next)
        q_prev = q_next
        # print(q_next[0] + q_next[1] + q_next[2])
        qdot_prev = q_dot_next
    
    q_traj = np.array(q_traj)
    qdot_traj = np.array(qdot_traj)

    return (q_traj,qdot_traj)

def joint_accl_traj(qdot_traj,dt):
    qddot_traj = []
    for i in range(len(qdot_traj)-1):
        # J = Jacob(q_traj[i],"2dim")
        # Jdot = Jacob_dot(q_traj[i],qdot_traj[i])
        # qddot = np.matmul(np.linalg.pinv(J),(np.array([xddot[i],yddot[i]]) - np.matmul(Jdot,qdot_traj[i])))
        qddot = (qdot_traj[i+1] - qdot_traj[i])/dt
        qddot_traj.append(qddot)
    qddot_traj.append(np.zeros(7))
    qddot_traj = np.array(qddot_traj)
    return qddot_traj

# Computing the Dynamics
def inverse_dynamics(M,C_qdot,g_vec,qddot,tauf):
    qddot = np.array(qddot).reshape(7,1)
    tau = np.matmul(M,qddot) + C_qdot + g_vec + tauf
    # print(qddot.shape)
    return tau

def forward_dynamics(M,C_qdot,g_vec,tau,tauf,q_prev,qdot_prev,dt):
    qddot = np.matmul(np.linalg.inv(M), tau - tauf - C_qdot - g_vec).reshape(7,1)
    # print(qddot.shape)
    q_new = q_prev + qdot_prev*dt
    qdot_new = qdot_prev + qddot*dt
    return (q_new,qdot_new)
    
# Define the Motion Control Loop
def motion_control(q_d_traj,qdot_d_traj,qddot_d_traj,q_init,qdot_init,dt):
    q_a = np.array(q_init).reshape(7,1)
    qdot_a = np.array(qdot_init).reshape(7,1)
    q_a_traj = [q_a]
    qdot_a_traj = [qdot_a]
    integral_q = np.zeros(7).reshape(7,1)
    prev_error_q = np.zeros(7).reshape(7,1)
    for i in range(len(q_d_traj)-1):
        print(i)
        # M_d = inertia_matrix(q_d_traj[i+1])
        # C_qdot_d = coriolis_vector(q_d_traj[i+1],qdot_d_traj[i+1])
        # g_vec_d = gravity_vec(q_d_traj[i+1])
        M_d = np.array(eng.get_MassMatrix(np.array(q_d_traj[i+1])))
        g_vec_d = np.array(eng.get_GravityVector(np.array(q_d_traj[i+1])))
        C_qdot_d = np.array(eng.get_CoriolisVector(np.array(q_d_traj[i+1]),np.array(qdot_d_traj[i+1])))
        tauf_d = np.array(eng.get_FrictionTorque(np.array(qdot_d_traj[i+1])))
        tau_d = inverse_dynamics(M_d,C_qdot_d,g_vec_d,np.array(qddot_d_traj[i+1]),tauf_d)

        # # Applying PID Control
        # [Kp,Ki,Kd] = [5, 0.1, 0.01]
        # # dt = (x_traj[i+1] - x_traj[i])/xdot_traj[i]
        # error_q = (q_d_traj[i]).reshape(7,1) - q_a
        # # print(error_q)
        # derivative_q = (error_q - prev_error_q) / dt
        # integral_q += error_q * dt
        # tau_pid = Kp * error_q + Ki * integral_q + Kd * derivative_q
        # prev_error_q = error_q
        # tau_pid = np.zeros(7)

        tau_total = tau_d
        # print(tau_total)

        # M_a = inertia_matrix(q_a)
        # C_qdot_a = coriolis_vector(q_a,qdot_a)
        # g_vec_a = gravity_vec(q_a)
        M_a = np.array(eng.get_MassMatrix(np.array(q_a)))
        g_vec_a = np.array(eng.get_GravityVector(np.array(q_a)))
        C_qdot_a = np.array(eng.get_CoriolisVector(np.array(q_a),np.array(qdot_a)))
        tauf_a = np.array(eng.get_FrictionTorque(np.array(qdot_a)))
        (q_a,qdot_a) = forward_dynamics(M_a,C_qdot_a,g_vec_a,tau_total,tauf_a,q_a,qdot_a,dt)
        # q_a = q_a.reshape(7,1)
        # qdot_a = qdot_a.reshape(7,1)
        print(q_a.shape)

        q_a_traj.append(q_a)
        qdot_a_traj.append(qdot_a)
    
    q_a_traj = np.array(q_a_traj)
    qdot_a_traj = np.array(qdot_a_traj)

    return (q_a_traj,qdot_a_traj)


# Check Dynamics
# def check_dyn(q_init,qdot_init,n,dt):
#     q = q_init
#     qdot = qdot_init
#     K_traj = []
#     V_traj = []
#     T_traj = []
#     for i in range(n):
#         M = inertia_matrix(q)
#         C_qdot = coriolis_vector(q,qdot)
#         g_vec = gravity_vec(q)
#         K = KineticEnergy(q,qdot)
#         V = PotentialEnergy(q)
#         T = K+V 
#         K_traj.append(K)
#         V_traj.append(V)
#         T_traj.append(T)
#         (q,qdot) = forward_dynamics(M,C_qdot,g_vec,0,np.array(q),np.array(qdot),dt)
#     return K_traj, V_traj, T_traj

# K_traj, V_traj, T_traj = check_dyn([np.pi/4, 0.1, 1.1], [0,0,0], 100, dt=0.01)
# Time_T = np.linspace(0,1,101)[:-1]

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 1, 1)
# plt.plot(Time_T, K_traj, label='Kinetic Energy (K)')
# plt.plot(Time_T, V_traj, label='Potential Energy (V)')
# plt.plot(Time_T, T_traj, label='Total Energy (T)')
# plt.title('Energy vs. Time')
# plt.legend()

# plt.tight_layout()
# plt.show()


# Straight Line Trajectory 
Type = ["min_norm", "min_joint_vel"]
num_points = 101
Total_Time = 1 # in seconds
t = np.linspace(0,Total_Time,num_points)
dt = t[1] - t[0] 

# x_lim = (l1+l2+l3)*sin(2*np.pi*t)
# y_lim = (l1+l2+l3)*cos(2*np.pi*t)

q_init = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
dh_params_init = DH_Parameters(q_init)

J_init = Jacob(dh_params_init)
X_init = forward_kin(dh_params_init)
[x0,y0,z0,phi0,theta0,psi0] = X_init
print(x0,y0,z0)
xf,yf,zf = -0.1,-0.2,z0
# print(x0,y0)
downscale = 10
x_traj = t/downscale + x0
# print(x_traj)
slope = (yf-y0)/(xf-x0)
y_traj = slope*x_traj+ y0-slope*x0
z_traj = z0*np.ones(num_points)
phi_traj = phi0*np.ones(num_points)
theta_traj = theta0*np.ones(num_points)
psi_traj = psi0*np.ones(num_points)

X_traj = []
for i in range(num_points):
    X_traj.append([x_traj[i],y_traj[i],z_traj[i],phi_traj[i],theta_traj[i],psi_traj[i]])
X_traj = np.array(X_traj)


# xdot_traj = np.array(np.ones(num_points-1).tolist() + [0])
# xdot_traj = np.array([0.001] + np.ones(num_points-2).tolist() + [0])
xdot_traj = (1/downscale)*np.array(np.ones(num_points))
ydot_traj = slope*xdot_traj
zdot_traj = np.zeros(num_points)
phidot_traj = np.zeros(num_points)
thetadot_traj = np.zeros(num_points)
psidot_traj = np.zeros(num_points)

Xdot_traj = []
for i in range(num_points):
    Xdot_traj.append([xdot_traj[i],ydot_traj[i],zdot_traj[i],phidot_traj[i],thetadot_traj[i],psidot_traj[i]])
Xdot_traj = np.array(Xdot_traj)


# t_act = 0
# t_actual = [t_act]
# xddot_traj = []
# for i in range(len(x_traj)-1):
#     dt_ = (x_traj[i+1] - x_traj[i])/xdot_traj[i]
#     xddot_traj.append((xdot_traj[i+1]-xdot_traj[i])/dt_)
#     t_act += dt_
#     t_actual.append(t_act)
# xddot_traj.append(0)
# xddot_traj = np.array(xddot_traj)
t_actual = t
xddot_traj = np.zeros(num_points)
yddot_traj = slope*xddot_traj
zddot_traj = np.zeros(num_points)
phiddot_traj = np.zeros(num_points)
thetaddot_traj = np.zeros(num_points)
psiddot_traj = np.zeros(num_points)

Xddot_traj = []
for i in range(num_points):
    Xddot_traj.append([xddot_traj[i],yddot_traj[i],zddot_traj[i],phiddot_traj[i],thetaddot_traj[i],psiddot_traj[i]])
Xddot_traj = np.array(Xddot_traj)

qdot_init = np.matmul(np.linalg.pinv(J_init),np.array(X_init))

# for i in range()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x_traj, y_traj, label='Desired xy traj')
# plt.plot(x_lim, y_lim, label='Workspace Limits')
plt.title('y vs. x')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_actual, x_traj, label='Desired x')
plt.plot(t_actual, y_traj, label='Desired y')
plt.plot(t_actual, z_traj, label='Desired z')
plt.plot(t_actual, phi_traj, label='Desired phi')
plt.plot(t_actual, theta_traj, label='Desired theta')
plt.plot(t_actual, psi_traj, label='Desired psi')
plt.title('X vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

# Main Desired Joint Trajectory Computation (You need to mention the desired scheme)
q_traj_desired = joint_traj(q_init,qdot_init,Xdot_traj,Type[0],dt, step_size=1.05)  # Input the type of optimization scheme in the function (argument = 'type')
qddot_traj_desired = joint_accl_traj(q_traj_desired[1],dt)
X_generated_desired_traj = []
# y_generated_desired_traj = []

Xdot_gen_traj = []
# ydot_gen_traj = []

for i in range(len(q_traj_desired[0])):
    dh_par = DH_Parameters(q_traj_desired[0][i])
    X_gen = forward_kin(dh_par)
    J_ = Jacob(dh_par)
    Xdot_gen = forward_vel_kin(J_,q_traj_desired[1][i]) 
    X_generated_desired_traj.append(X_gen)
    # y_generated_desired_traj.append(y_gen)
    Xdot_gen_traj.append(Xdot_gen)
    # ydot_gen_traj.append(ydot_gen)

X_generated_desired_traj = np.array(X_generated_desired_traj)
Xdot_gen_traj = np.array(Xdot_gen_traj)

plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(X_generated_desired_traj[:,0], X_generated_desired_traj[:,1], label='Desired generated xy traj')
plt.plot(X_traj[:,0], X_traj[:,1], label='Desired xy traj', linestyle='--')
plt.title('y vs. x')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(t_actual, X_generated_desired_traj[:,0], label='Desired gen x')
plt.plot(t_actual, X_traj[:,0], label='Desired x', linestyle='--')
plt.title('x vs. Time')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(t_actual, X_generated_desired_traj[:,1], label='Desired gen y')
plt.plot(t_actual, X_traj[:,1], label='Desired y', linestyle='--')
plt.title('y vs. Time')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(t_actual, X_generated_desired_traj[:,2], label='Desired gen z')
plt.plot(t_actual, X_traj[:,2], label='Desired z', linestyle='--')
plt.title('z vs. Time')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(t_actual, X_generated_desired_traj[:,3], label='Desired gen phi')
plt.plot(t_actual, X_traj[:,3], label='Desired phi', linestyle='--')
plt.title('phi vs. Time')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(t_actual, X_generated_desired_traj[:,4], label='Desired gen theta')
plt.plot(t_actual, X_traj[:,4], label='Desired theta', linestyle='--')
plt.title('theta vs. Time')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(t_actual, X_generated_desired_traj[:,5], label='Desired gen psi')
plt.plot(t_actual, X_traj[:,5], label='Desired psi', linestyle='--')
plt.title('psi vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(X_generated_desired_traj[:,0], X_generated_desired_traj[:,1], label='Desired generated xy traj')
plt.plot(X_traj[:,0], X_traj[:,1], label='Desired xy traj', linestyle='--')
plt.title('y vs. x')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(t_actual, Xdot_gen_traj[:,0], label='Desired gen xdot')
plt.plot(t_actual, Xdot_traj[:,0], label='Desired xdot', linestyle='--')
plt.title('xdot vs. Time')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(t_actual, Xdot_gen_traj[:,1], label='Desired gen ydot')
plt.plot(t_actual, Xdot_traj[:,1], label='Desired ydot', linestyle='--')
plt.title('ydot vs. Time')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(t_actual, Xdot_gen_traj[:,2], label='Desired gen zdot')
plt.plot(t_actual, Xdot_traj[:,2], label='Desired zdot', linestyle='--')
plt.title('zdot vs. Time')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(t_actual, Xdot_gen_traj[:,3], label='Desired gen phidot')
plt.plot(t_actual, Xdot_traj[:,3], label='Desired phidot', linestyle='--')
plt.title('phidot vs. Time')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(t_actual, Xdot_gen_traj[:,4], label='Desired gen thetadot')
plt.plot(t_actual, Xdot_traj[:,4], label='Desired thetadot', linestyle='--')
plt.title('thetadot vs. Time')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(t_actual, Xdot_gen_traj[:,5], label='Desired gen psidot')
plt.plot(t_actual, Xdot_traj[:,5], label='Desired psidot', linestyle='--')
plt.title('psidot vs. Time')
plt.legend()

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(t_actual, q_traj_desired[0][:,0], label='Desired generated q1 traj')
plt.plot(t_actual, q_traj_desired[0][:,1], label='Desired generated q2 traj')
plt.plot(t_actual, q_traj_desired[0][:,2], label='Desired generated q3 traj')
plt.plot(t_actual, q_traj_desired[0][:,3], label='Desired generated q4 traj')
plt.plot(t_actual, q_traj_desired[0][:,4], label='Desired generated q5 traj')
plt.plot(t_actual, q_traj_desired[0][:,5], label='Desired generated q6 traj')
plt.plot(t_actual, q_traj_desired[0][:,6], label='Desired generated q7 traj')
plt.title('q_desired vs. t')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(t_actual, q_traj_desired[1][:,0], label='Desired generated q1dot traj')
plt.plot(t_actual, q_traj_desired[1][:,1], label='Desired generated q2dot traj')
plt.plot(t_actual, q_traj_desired[1][:,2], label='Desired generated q3dot traj')
plt.plot(t_actual, q_traj_desired[1][:,3], label='Desired generated q4dot traj')
plt.plot(t_actual, q_traj_desired[1][:,4], label='Desired generated q5dot traj')
plt.plot(t_actual, q_traj_desired[1][:,5], label='Desired generated q6dot traj')
plt.plot(t_actual, q_traj_desired[1][:,6], label='Desired generated q7dot traj')
plt.title('qdot_desired vs. Time')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(t_actual, qddot_traj_desired[:,0], label='Desired generated q1ddot traj')
plt.plot(t_actual, qddot_traj_desired[:,1], label='Desired generated q2ddot traj')
plt.plot(t_actual, qddot_traj_desired[:,2], label='Desired generated q3ddot traj')
plt.plot(t_actual, qddot_traj_desired[:,3], label='Desired generated q4ddot traj')
plt.plot(t_actual, qddot_traj_desired[:,4], label='Desired generated q5ddot traj')
plt.plot(t_actual, qddot_traj_desired[:,5], label='Desired generated q6ddot traj')
plt.plot(t_actual, qddot_traj_desired[:,6], label='Desired generated q7ddot traj')
plt.title('qddot vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

(q_actual_traj,qdot_actual_traj) = motion_control(q_traj_desired[0],q_traj_desired[1],qddot_traj_desired,q_init,qdot_init,dt)

X_actual_traj = []
# y_actual_traj = []
Xdot_actual_traj = []
# ydot_actual_traj = []

for i in range(len(q_actual_traj)):
    dh_par_actual = DH_Parameters(q_traj_desired[0][i])
    X_actual = forward_kin(dh_par_actual)
    X_actual_traj.append(X_actual)
    # y_actual_traj.append(y_actual)
    J_actual = Jacob(dh_par_actual)
    Xdot_actual = forward_vel_kin(J_actual,np.array(qdot_actual_traj[i]))
    Xdot_actual_traj.append(Xdot_actual)
    # ydot_actual_traj.append(ydot_actual)
X_actual_traj = np.array(X_actual_traj)
Xdot_actual_traj = np.array(Xdot_actual_traj)

plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(t_actual, q_actual_traj[:,0], label='Actual q1 traj')
plt.plot(t_actual, q_traj_desired[0][:,0], label='Desired q1 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(t_actual, q_actual_traj[:,1], label='Actual q2 traj')
plt.plot(t_actual, q_traj_desired[0][:,1], label='Desired q2 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(t_actual, q_actual_traj[:,2], label='Actual q3 traj')
plt.plot(t_actual, q_traj_desired[0][:,2], label='Desired q3 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(t_actual, q_actual_traj[:,3], label='Actual q4 traj')
plt.plot(t_actual, q_traj_desired[0][:,3], label='Desired q4 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(t_actual, q_actual_traj[:,4], label='Actual q5 traj')
plt.plot(t_actual, q_traj_desired[0][:,4], label='Desired q5 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(t_actual, q_actual_traj[:,5], label='Actual q6 traj')
plt.plot(t_actual, q_traj_desired[0][:,5], label='Desired q6 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(t_actual, q_actual_traj[:,6], label='Actual q7 traj')
plt.plot(t_actual, q_traj_desired[0][:,6], label='Desired q7 traj', linestyle='--')
plt.title('q vs. t')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(t_actual, qdot_actual_traj[:,0], label='Actual q1dot traj')
plt.plot(t_actual, q_traj_desired[1][:,0], label='Desired q1dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(t_actual, qdot_actual_traj[:,1], label='Actual q2dot traj')
plt.plot(t_actual, q_traj_desired[1][:,1], label='Desired q2dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(t_actual, qdot_actual_traj[:,2], label='Actual q3dot traj')
plt.plot(t_actual, q_traj_desired[1][:,2], label='Desired q3dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(t_actual, qdot_actual_traj[:,3], label='Actual q4dot traj')
plt.plot(t_actual, q_traj_desired[1][:,3], label='Desired q4dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(t_actual, qdot_actual_traj[:,4], label='Actual q5dot traj')
plt.plot(t_actual, q_traj_desired[1][:,4], label='Desired q5dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(t_actual, qdot_actual_traj[:,5], label='Actual q6dot traj')
plt.plot(t_actual, q_traj_desired[1][:,5], label='Desired q6dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(t_actual, qdot_actual_traj[:,6], label='Actual q7dot traj')
plt.plot(t_actual, q_traj_desired[1][:,6], label='Desired q7dot traj', linestyle='--')
plt.title('qdot vs. t')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.plot(X_actual_traj[:,0], X_actual_traj[:,1], label='Actual XY traj')
plt.plot(X_traj[:,0], X_traj[:,1], label='Desired XY traj', linestyle='--')
plt.title('Y vs. X')
plt.legend()

plt.subplot(2, 4, 2)
plt.plot(t_actual, X_actual_traj[:,0], label='Actual x')
plt.plot(t_actual, X_traj[:,0], label='Desired x', linestyle='--')
plt.title('x vs. Time')
plt.legend()

plt.subplot(2, 4, 3)
plt.plot(t_actual, X_actual_traj[:,1], label='Actual y')
plt.plot(t_actual, X_traj[:,1], label='Desired y', linestyle='--')
plt.title('y vs. Time')
plt.legend()

plt.subplot(2, 4, 4)
plt.plot(t_actual, X_actual_traj[:,2], label='Actual z')
plt.plot(t_actual, X_traj[:,2], label='Desired z', linestyle='--')
plt.title('z vs. Time')
plt.legend()

plt.subplot(2, 4, 5)
plt.plot(t_actual, X_actual_traj[:,3], label='Actual phi')
plt.plot(t_actual, X_traj[:,3], label='Desired phi', linestyle='--')
plt.title('phi vs. Time')
plt.legend()

plt.subplot(2, 4, 6)
plt.plot(t_actual, X_actual_traj[:,4], label='Actual theta')
plt.plot(t_actual, X_traj[:,4], label='Desired theta', linestyle='--')
plt.title('theta vs. Time')
plt.legend()

plt.subplot(2, 4, 7)
plt.plot(t_actual, X_actual_traj[:,5], label='Actual psi')
plt.plot(t_actual, X_traj[:,5], label='Desired psi', linestyle='--')
plt.title('psi vs. Time')
plt.legend()

plt.tight_layout()
plt.show()

plt.subplot(2, 3, 1)
plt.plot(t_actual, Xdot_actual_traj[:,0], label='Actual xdot')
plt.plot(t_actual, Xdot_traj[:,0], label='Desired xdot', linestyle='--')
plt.title('xdot vs. Time')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(t_actual, Xdot_actual_traj[:,1], label='Actual ydot')
plt.plot(t_actual, Xdot_traj[:,1], label='Desired ydot', linestyle='--')
plt.title('ydot vs. Time')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(t_actual, Xdot_actual_traj[:,2], label='Actual zdot')
plt.plot(t_actual, Xdot_traj[:,2], label='Desired zdot', linestyle='--')
plt.title('zdot vs. Time')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(t_actual, Xdot_actual_traj[:,3], label='Actual phidot')
plt.plot(t_actual, Xdot_traj[:,3], label='Desired phidot', linestyle='--')
plt.title('phidot vs. Time')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(t_actual, Xdot_actual_traj[:,4], label='Actual thetadot')
plt.plot(t_actual, Xdot_traj[:,4], label='Desired thetadot', linestyle='--')
plt.title('thetadot vs. Time')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(t_actual, Xdot_actual_traj[:,5], label='Actual psidot')
plt.plot(t_actual, Xdot_traj[:,5], label='Desired psidot', linestyle='--')
plt.title('psidot vs. Time')
plt.legend()

plt.tight_layout()
plt.show()