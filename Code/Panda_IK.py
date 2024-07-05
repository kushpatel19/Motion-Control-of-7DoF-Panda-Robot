import numpy as np
import math

def DH_Transform(a, d, alpha, theta):
    """
    Compute a single transformation matrix using DH parameters.
    """
    # Define the DH transformation matrix using the provided parameters.
    return np.array([[np.cos(theta),-np.sin(theta), 0, a],
                [np.sin(theta)*np.cos(alpha), np.cos(theta)*np.cos(alpha), -np.sin(alpha), -np.sin(alpha)*d],
                [np.sin(theta)*np.sin(alpha), np.cos(theta)*np.sin(alpha),  np.cos(alpha),  np.cos(alpha)*d],
                [   0,  0,  0,  1]])

def franka_IK_EE(O_T_EE_array, q7, q_actual_array):
    # Create arrays to hold NaN values.
    q_all_NAN = np.full((4, 7), np.nan)
    q_NAN = np.array([np.nan]*7)
    q_all = np.copy(q_all_NAN)
    
    O_T_EE = O_T_EE_array
    
    d1 = 0.333
    d3 = 0.316
    d5 = 0.384
    # d7e = 0.2104
    d7e = 0.107
    a4 = 0.0825
    a5 = -0.0825
    a7 = 0.088
    
    LL24 = 0.10666225  # a4^2 + d3^2
    LL46 = 0.15426225  # a5^2 + d5^2
    L24 = 0.326591870689  # sqrt(LL24)
    L46 = 0.392762332715  # sqrt(LL46)
    
    thetaH46 = 1.35916951803  # atan(d5/a4)
    theta342 = 1.31542071191  # atan(d3/a4)
    # thetaH46 = math.atan2(d5,a4)
    # theta342 = math.atan2(d3,a4)
    theta46H = 0.211626808766  # acot(d5/a4)
    
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    if q7 <= q_min[6] or q7 >= q_max[6]:
        return q_all_NAN
    else:
        q_all[:, 6] = q7

    # Extract components from the homogeneous transformation matrix.
    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    p_7 = p_EE - d7e * z_EE

    # Calculate the position and orientation of Frame 6 (p_6) and direction of the end effector's X-axis (x_6).
    x_EE_6 = np.array([math.cos(q7), -math.sin(q7), 0.0])
    x_6 = np.dot(R_EE, x_EE_6)
    x_6 /= np.linalg.norm(x_6)
    p_6 = p_7 - a7 * x_6

    p_2 = np.array([0.0, 0.0, d1])
    V26 = p_6 - p_2
    
    LL26 = np.sum(V26**2)
    L26 = math.sqrt(LL26)
    
    if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
        return q_all_NAN
    
    theta246 = math.acos((LL24 + LL46 - LL26) / (2.0 * L24 * L46))
    q4 = theta246 + thetaH46 + theta342 - 2.0 * math.pi
    q4_ = -(theta246 - thetaH46 - theta342)


    if q4 < q_min[3] or q4 > q_max[3]:
        return q_all_NAN
    else:
        q_all[:, 3] = q4
        # print(q_all)
    
    # compute q6
    theta462 = math.acos((LL26 + LL46 - LL24) / (2.0 * L26 * L46))
    theta26H = theta46H + theta462
    D26 = -L26 * math.cos(theta26H)

    Z_6 = np.cross(z_EE, x_6)
    Y_6 = np.cross(Z_6, x_6)
    R_6 = np.zeros((3, 3))
    R_6[:, 0] = x_6
    R_6[:, 1] = Y_6 / np.linalg.norm(Y_6)
    R_6[:, 2] = Z_6 / np.linalg.norm(Z_6)
    V_6_62 = R_6.T @ (-V26) ################### doubtfull

    Phi6 = math.atan2(V_6_62[1], V_6_62[0])
    Theta6 = math.asin(D26 / math.sqrt(V_6_62[0]**2 + V_6_62[1]**2))

    q6 = [math.pi - Theta6 - Phi6, Theta6 - Phi6]

    for i in range(2):
        if q6[i] < q_min[5]:
            q6[i] += 2.0 * math.pi
        elif q6[i] > q_max[5]:
            q6[i] -= 2.0 * math.pi

        if q6[i] < q_min[5] or q6[i] > q_max[5]:
            q_all[2*i] = q_NAN
            q_all[2*i + 1] = q_NAN
        else:
            q_all[2*i][5] = q6[i]
            q_all[2*i + 1][5] = q6[i]

    if math.isnan(q_all[0][5]) and math.isnan(q_all[2][5]):
        return q_all_NAN
    
    # compute q1 & q2
    thetaP26 = 3.0 * math.pi / 2 - theta462 - theta246 - theta342
    thetaP = math.pi - thetaP26 - theta26H
    LP6 = L26 * math.sin(thetaP26) / math.sin(thetaP)

    z_5_all = [np.zeros(3) for _ in range(4)]
    V2P_all = [np.zeros(3) for _ in range(4)]

    for i in range(2):
        z_6_5 = np.array([math.sin(q6[i]), math.cos(q6[i]), 0.0])
        z_5 = R_6 @ z_6_5
        V2P = V26 - LP6 * z_5

        z_5_all[2 * i] = z_5
        z_5_all[2 * i + 1] = z_5
        V2P_all[2 * i] = V2P
        V2P_all[2 * i + 1] = V2P

        L2P = np.linalg.norm(V2P)

        if abs(V2P[2] / L2P) > 0.9999999:              # Singularity Resolution
            q_all[2 * i][0] = q_actual_array[0]
            q_all[2 * i][1] = 0.0
            q_all[2 * i + 1][0] = q_actual_array[0]
            q_all[2 * i + 1][1] = 0.0
        else:
            q_all[2 * i][0] = math.atan2(V2P[1], V2P[0])
            q_all[2 * i][1] = math.acos(V2P[2] / L2P)
            # if q_all[2 * i][0] < 0:
            #     q_all[2 * i + 1][0] = q_all[2 * i][0] + math.pi
            # else:
            #     q_all[2 * i + 1][0] = q_all[2 * i][0] - math.pi
            q_all[2 * i + 1][0] = math.atan2(-V2P[1], -V2P[0])
            q_all[2 * i + 1][1] = -q_all[2 * i][1]

    for i in range(4):
        if (q_all[i][0] < q_min[0] or q_all[i][0] > q_max[0] or q_all[i][1] < q_min[1] or q_all[i][1] > q_max[1]):
            q_all[i] = q_NAN 

        # compute q3
        z_3 = V2P_all[i] / np.linalg.norm(V2P_all[i])
        Y_3 = -np.cross(V26, V2P_all[i])
        y_3 = Y_3 / np.linalg.norm(Y_3)
        x_3 = np.cross(y_3, z_3)
        # R_1 = np.zeros((3, 3))
        # c1 = np.cos(q_all[i][0])
        # s1 = np.sin(q_all[i][0])
        # R_1[0, :] = [c1, -s1, 0.0]
        # R_1[1, :] = [s1,  c1, 0.0]
        # R_1[2, :] = [0.0, 0.0, 1.0]
        R_1 = DH_Transform(0,d1,0,q_all[i][0])[:3, :3]
        # R_1_2 = np.zeros((3, 3))
        # c2 = np.cos(q_all[i][1])
        # s2 = np.sin(q_all[i][1])
        # R_1_2[0, :] = [c2, -s2, 0.0]  ######### Doubtfull
        # R_1_2[1, :] = [0.0, 0.0, 1.0]
        # R_1_2[2, :] = [-s2, -c2, 0.0]
        R_1_2 = DH_Transform(0,0,-np.pi/2,q_all[i][1])[:3, :3]
        R_2 = R_1 @ R_1_2
        x_2_3 = R_2.T @ x_3
        q_all[i][2] = np.arctan2(x_2_3[2], x_2_3[0])

        if q_all[i][2] < q_min[2] or q_all[i][2] > q_max[2]:
            q_all[i] = q_NAN       

        # compute q5
        VH4 = p_2 + d3 * z_3 + a4 * x_3 - p_6 + d5 * z_5_all[i]
        # R_5_6 = np.zeros((3, 3))
        # c6 = np.cos(q_all[i][5])
        # s6 = np.sin(q_all[i][5])
        # R_5_6[0, :] = [c6, -s6, 0.0]   ######## Doubtfull
        # R_5_6[1, :] = [0.0, 0.0, -1.0]
        # R_5_6[2, :] = [s6, c6, 0.0]
        R_5_6 = DH_Transform(0,0,np.pi/2,q_all[i][5])[:3, :3]
        R_5 = R_6 @ R_5_6.T
        V_5_H4 = R_5.T @ VH4
        q_all[i][4] = -np.arctan2(V_5_H4[1], V_5_H4[0])

        if q_all[i][4] < q_min[4] or q_all[i][4] > q_max[4]:
            q_all[i] = q_NAN

    return q_all

# # Now you can call the function and test it:
# O_T_EE_array = np.array([[0,1,0,0],[0,0,1,0],[1,0,0,0],[0,0,0,1]]) # Fill in the data
# q7 = -np.pi/4              # Fill in the value
# q_actual_array = [0,0,0,-0.2,0,0,-np.pi/4]   # Fill in the data

# results = franka_IK_EE(O_T_EE_array, q7, q_actual_array)
# print(results)
