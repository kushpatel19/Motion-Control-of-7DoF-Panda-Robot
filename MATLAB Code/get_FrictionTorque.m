function tau_F = get_FrictionTorque(qdot)

    fp = [...
        0.54615,0.87224,0.64068,1.2794,0.83904,0.30301,0.56489;...
        0,0,0,0,0,0,0;...
        0.039533,0.025882,-0.04607,0.036194,0.026226,-0.021047,0.0035526;...
        5.1181,9.0657,10.136,5.5903,8.3469,17.133,10.336];

    A = fp(1,:);
    k = fp(2,:);
    qdotsign = fp(3,:);
    alpha= fp(4,:);
    
    num_of_joints = length(qdot);
    tau_F = zeros(num_of_joints,1);
    
    for i=1:num_of_joints
        tau_F(i) = -A(i)/(1+exp(-alpha(i)*qdotsign(i))) + A(i)/(1+exp(-alpha(i)*(qdot(i)+qdotsign(i))));
    end