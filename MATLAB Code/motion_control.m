clear all
close all
clc

t_in = 0; % [s]
t_fin = 1; % [s]
delta_t = 0.01; % [s]

num_of_joints = 7;

t = t_in:delta_t:t_fin;

Q = zeros(num_of_joints,length(t));
dQ = zeros(num_of_joints,length(t));
ddQ = zeros(num_of_joints,length(t));
TAU_FF = zeros(num_of_joints,length(t));

Kp = 1;
Kd = 0.01;

q_actual = zeros(num_of_joints, length(t));
dq_actual = zeros(num_of_joints, length(t));
ddq_actual = zeros(num_of_joints, length(t));
q_err = zeros(num_of_joints, length(t));
dq_err = zeros(num_of_joints, length(t));

for j=1:num_of_joints
    Q(j,:) = sin(t);
    dQdt(j,:) = cos(t);
    ddQdt(j,:) = -sin(t);
end

% Initialize plot
figure;
plotHandle = plot3(0, 0, 0, 'LineWidth', 2);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('End-Effector Trajectory');
grid on;
axis equal;

% Preallocate arrays for animation
x = zeros(1, length(t));
y = zeros(1, length(t));
z = zeros(1, length(t));
phi = zeros(1, length(t));
theta = zeros(1, length(t));
psi = zeros(1, length(t));

% Preallocate arrays for joint trajectory plots
q_plot = zeros(num_of_joints, length(t));
q_actual_plot = zeros(num_of_joints, length(t));

for i=1:length(t)
    q(:,i) = Q(:,i);
    dq(:,i) = dQdt(:,i);
    ddq(:,i) = ddQdt(:,i);
    
    q_actual(:,1) = q(:,1);
    dq_actual(:,1) = dq(:,1);
    
    gd = get_GravityVector(q(:,i));
    cd = get_CoriolisVector(q(:,i),dq(:,i));
    Md = get_MassMatrix(q(:,i));
    taufd = get_FrictionTorque(dq(:,i));
    
    g = get_GravityVector(q_actual(:,i));
    c = get_CoriolisVector(q_actual(:,i),dq_actual(:,i));
    M = get_MassMatrix(q_actual(:,i));
    tauf = get_FrictionTorque(dq_actual(:,i));
    
    q_err(:,i) = q(:,i) - q_actual(:,i);
    dq_err(:,i) = dq(:,i) - dq_actual(:,i);

    TAU_FF(:,i) = Md*ddq(:,i) + cd + gd + taufd;
    TAU_FB(:,i) = Kp*q_err(:,i) + Kd*dq_err(:,i);
    
    % Check for singularity
    lambda = 1e-5; % Small positive value
    if cond(M) < 1e-10
        warning('Matrix is close to singular. Handle singularity case.');
        ddq_actual(:, i) = pinv(M + lambda * eye(size(M))) * ((TAU_FF(:, i) + TAU_FB(:, i)) - (c + g + tauf)); % With PD Control
    else
        ddq_actual(:, i) = (M + lambda * eye(size(M))) \ ((TAU_FF(:, i) + TAU_FB(:, i)) - (c + g + tauf)); % With PD Control
    end
    
%     ddq_actual(:,i) = (M)\((TAU_FF(:,i))-(c+g+tauf));           % Without Control
%     ddq_actual(:,i) = (M + lambda * eye(size(M)))\((TAU_FF(:,i)+TAU_FB(:,i))-(c+g+tauf)); % With PD Control

    dq_actual(:,i+1) = dq_actual(:,i) + ddq_actual(:,i)*delta_t;

    q_actual(:,i+1) = q_actual(:,i) + dq_actual(:,i)*delta_t + 0.5*ddq_actual(:,i)*delta_t^2;

    [x(i), y(i), z(i), phi(i), theta(i), psi(i)] = FK_Panda(q(:,i));

    % Update plot
    set(plotHandle, 'XData', x(1:i), 'YData', y(1:i), 'ZData', z(1:i));
    drawnow;

    % Pause to control animation speed
    pause(delta_t);
end

%%
figure
for j=1:num_of_joints
    subplot(4,2,j);
    plot(t,TAU_FF(j,:))
    xlabel('time [s]');
    ylabeltext = sprintf('_%i [Nm]',j);
    ylabel(['\tau' ylabeltext]);
    grid;
end

%% Plotting joint trajectories
figure;

for j = 1:num_of_joints
    subplot(num_of_joints, 1, j);
    plot(t, q(j, 1:i), t, q_actual(j, 1:i), '--');
    xlabel('time [s]');
    ylabel(['q_' num2str(j)]);
    legend(['Desired q_' num2str(j)], ['Actual q_' num2str(j)]);
    grid;
end
% 
% %% Plotting
% figure;
% 
% % Plot joint trajectory q
% subplot(3, 1, 1);
% plot(t, Q);
% xlabel('time [s]');
% ylabel('q');
% legend('q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'q_7');
% grid;
% 
% %% Plot joint velocity q_dot
% subplot(3, 1, 2);
% plot(t, dQdt);
% xlabel('time [s]');
% ylabel('q\_dot');
% legend('q\_dot\_1', 'q\_dot\_2', 'q\_dot\_3', 'q\_dot\_4', 'q\_dot\_5', 'q\_dot\_6', 'q\_dot\_7');
% grid;
% 
% % Plot joint acceleration q_ddot
% subplot(3, 1, 3);
% plot(t, ddQdt);
% xlabel('time [s]');
% ylabel('q\_ddot');
% legend('q\_ddot\_1', 'q\_ddot\_2', 'q\_ddot\_3', 'q\_ddot\_4', 'q\_ddot\_5', 'q\_ddot\_6', 'q\_ddot\_7');
% grid;



function [x, y, z, phi, theta, psi] = FK_Panda(q)
    dh_parameters = [
        0, 0.333, 0, q(1);
        0, 0, -pi/2, q(2);
        0, 0.316, pi/2, q(3);
        0.0825, 0, pi/2, q(4);
        -0.0825, 0.384, -pi/2, q(5);
        0, 0, pi/2, q(6);
        0.088, 0, pi/2, q(7);
        0, 0.107, 0, 0
    ];

    T_final = eye(4);

    for i = 1:size(dh_parameters, 1)
        T = dh_transform(dh_parameters(i, 1), dh_parameters(i, 2), dh_parameters(i, 3), dh_parameters(i, 4));
        T_final = T_final * T;
    end

    x = T_final(1, 4);
    y = T_final(2, 4);
    z = T_final(3, 4);

    R = T_final(1:3, 1:3);
    [phi, theta, psi] = rotation_matrix_to_euler_angles(R);
end

function T = dh_transform(a, d, alpha, theta)
    T = [cos(theta), -sin(theta), 0, a;
         sin(theta) * cos(alpha), cos(theta) * cos(alpha), -sin(alpha), -d * sin(alpha);
         sin(theta) * sin(alpha), cos(theta) * sin(alpha), cos(alpha), d * cos(alpha);
         0, 0, 0, 1];
end

function [phi, theta, psi] = rotation_matrix_to_euler_angles(R)
    psi = atan2(R(2, 1), R(1, 1));
    theta = atan2(-R(3, 1), sqrt(R(3, 2)^2 + R(3, 3)^2));
    phi = atan2(R(3, 2), R(3, 3));
end

function numerical_jacobian = computeNumericalJacobian(q)
    % Create numerical DH parameters
    dh_parameters = [
        [0, 0.333, 0, q(1)];
        [0, 0, -pi/2, q(2)];
        [0, 0.316, pi/2, q(3)];
        [0.0825, 0, pi/2, q(4)];
        [-0.0825, 0.384, -pi/2, q(5)];
        [0, 0, pi/2, q(6)];
        [0.088, 0, pi/2, q(7)];
        [0, 0.107, 0, 0]
    ];

    % Initialize the Jacobian matrix
    num_joints = size(dh_parameters, 1) - 1;
    numerical_jacobian = zeros(6, num_joints);
    T = eye(4);

    for i = 1:num_joints + 1
        a = dh_parameters(i, 1);
        d = dh_parameters(i, 2);
        alpha = dh_parameters(i, 3);
        theta = dh_parameters(i, 4);

        % Create the transformation matrix
        T_i_i = [
            cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta);
            sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta);
            0, sin(alpha), cos(alpha), d;
            0, 0, 0, 1
        ];

        T = T * T_i_i;
    end

    p_e = T(1:3, 4);
    T = eye(4);

    for i = 1:num_joints
        a = dh_parameters(i, 1);
        d = dh_parameters(i, 2);
        alpha = dh_parameters(i, 3);
        theta = dh_parameters(i, 4);

        % Create the transformation matrix
        T_i = [
            cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta);
            sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta);
            0, sin(alpha), cos(alpha), d;
            0, 0, 0, 1
        ];

        T = T * T_i;
        p = T(1:3, 4);
        z = T(1:3, 3);

        % Calculate linear and angular velocity parts of the Jacobian matrix
        numerical_jacobian(1:3, i) = cross(z, p_e - p);
        numerical_jacobian(4:6, i) = z;
    end
end

function sym_jacobian = computeSymbolicJacobian(q)
    % Define symbolic joint angles
    syms q1 q2 q3 q4 q5 q6 q7
    
    % Create symbolic DH parameters
    dh_parameters = [
        [0, 0.333, 0, q1];
        [0, 0, -pi/2, q2];
        [0, 0.316, pi/2, q3];
        [0.0825, 0, pi/2, q4];
        [-0.0825, 0.384, -pi/2, q5];
        [0, 0, pi/2, q6];
        [0.088, 0, pi/2, q7];
        [0, 0.107, 0, 0]
    ];

    % Compute the symbolic Jacobian
    num_joints = size(dh_parameters, 1) - 1;
    sym_jacobian = sym(zeros(6, num_joints));
    T = eye(4);

    for i = 1:num_joints + 1
        a = dh_parameters(i, 1);
        d = dh_parameters(i, 2);
        alpha = dh_parameters(i, 3);
        theta = dh_parameters(i, 4);

        % Create the transformation matrix
        T_i_i = [
            cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta);
            sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta);
            0, sin(alpha), cos(alpha), d;
            0, 0, 0, 1
        ];

        T = T * T_i_i;
    end

    p_e = T(1:3, 4);
    T = eye(4);

    for i = 1:num_joints
        a = dh_parameters(i, 1);
        d = dh_parameters(i, 2);
        alpha = dh_parameters(i, 3);
        theta = dh_parameters(i, 4);

        % Create the transformation matrix
        T_i = [
            cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta);
            sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta);
            0, sin(alpha), cos(alpha), d;
            0, 0, 0, 1
        ];

        T = T * T_i;
        p = T(1:3, 4);
        z = T(1:3, 3);

        % Calculate linear and angular velocity parts of the Jacobian matrix
        sym_jacobian(1:3, i) = cross(z, p_e - p);
        sym_jacobian(4:6, i) = z;
    end
end
