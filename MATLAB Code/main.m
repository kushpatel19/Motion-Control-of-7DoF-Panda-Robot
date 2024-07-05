
clear all
close all
clc

t_in = 0; % [s]
t_fin = 5; % [s]
delta_t = 0.01; % [s]

num_of_joints = 7;

t = t_in:delta_t:t_fin;

Q = zeros(num_of_joints,length(t));
dQ = zeros(num_of_joints,length(t));
ddQ = zeros(num_of_joints,length(t));
TAU = zeros(num_of_joints,length(t));

for j=1:num_of_joints
    Q(j,:) = sin(t);
    dQdt(j,:) = cos(t);
    ddQdt(j,:) = -sin(t);
end

for i=1:length(t)
    q = Q(:,i);
    dq = dQdt(:,i);
    ddq = ddQdt(:,i);
    
    g = get_GravityVector(q);
    c = get_CoriolisVector(q,dq);
    M = get_MassMatrix(q);
    tauf = get_FrictionTorque(dq);
    
    TAU(:,i) = M*ddq + c + g + tauf;
    
    % equivalently, you could use (decomment) the following two lines:
%     Cmat = get_CoriolisMatrix(q,dq);
%     TAU(:,i) = M*ddq + Cmat*dq + g + tauf;
end

figure
for j=1:num_of_joints
    subplot(4,2,j);
    plot(t,TAU(j,:))
    xlabel('time [s]');
    ylabeltext = sprintf('_%i [Nm]',j);
    ylabel(['\tau' ylabeltext]);
    grid;
end

% % Plotting
% figure;
% 
% % Plot joint trajectory q
% subplot(4, 1, 1);
% plot(t, Q);
% xlabel('time [s]');
% ylabel('q');
% legend('q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6', 'q_7');
% grid;
% 
% % Plot joint velocity q_dot
% subplot(4, 1, 2);
% plot(t, dQdt);
% xlabel('time [s]');
% ylabel('q\_dot');
% legend('q\_dot\_1', 'q\_dot\_2', 'q\_dot\_3', 'q\_dot\_4', 'q\_dot\_5', 'q\_dot\_6', 'q\_dot\_7');
% grid;
% 
% % Plot joint acceleration q_ddot
% subplot(4, 1, 3);
% plot(t, ddQdt);
% xlabel('time [s]');
% ylabel('q\_ddot');
% legend('q\_ddot\_1', 'q\_ddot\_2', 'q\_ddot\_3', 'q\_ddot\_4', 'q\_ddot\_5', 'q\_ddot\_6', 'q\_ddot\_7');
% grid;
% 
% % Plot joint torque TAU
% subplot(4, 1, 4);
% plot(t, TAU);
% xlabel('time [s]');
% ylabel('\tau [Nm]');
% legend('\tau_1', '\tau_2', '\tau_3', '\tau_4', '\tau_5', '\tau_6', '\tau_7');
% grid;