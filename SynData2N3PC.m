% generate synthetic data for 2 neurons

% define parameters
L   = 3;                 % number of pc kept   
T   = 6000;              % experiment length
N   = 2;                 % total number of cells   
Fs  = 5000;              % sampling frequency in Hz
J   = 2;                 % total number of clusters
R   = 2;                 % observation noise
Q   = 0.001;             % process noise 
r01 = 1;                 % transition rate from spike state to r.p. state
r12 = 0.5;               % transition rate from r.p. state to isi state
r20 = [0.2  0.1];        % transition rate from isi state to spike state neuron 1 and 2              
no_obs = 100*ones(1,L);  % missing-observation symbol

Pa_1 = [1-r01 r01 0; 0 1-r12 r12; r20(1) 0 1-r20(1)]; % hmm transition matrix neuron 1
Pa_2 = [1-r01 r01 0; 0 1-r12 r12; r20(2) 0 1-r20(2)]; % hmm transition matrix neuron 2

% three-state model for the neuron cluster
    % 0: 'spike state' cluster produces a spike with p = 1;
    % 1: 'refractory period' cluster incapable of producing a spike, p = 0;
    % 2: 'isi state' p = 0; 
    
% auxiliary structures
spkt   = zeros(1,T);
obs_id = zeros(1, T);

% initial state of of the system:
%   at t = 1, both cells are in state 2 (isi state) and Y(1) = no_obs.
q(1, 1) = 2;    
q(2, 1) = 2;
V(1, :) = no_obs;
cluster_id(1) = -1;
obs_id(1)     = -1; 
    
% generate Markov process for each cell
for t = 1 : T-1
    rn = rand;
    for i = 1 : N
        if q(i, t) == 0;
            if rn < r01
                q(i, t+1) = 1;
            else
                q(i, t+1) = 0;
            end
        end
        if q(i, t) == 1
            if rn > r12
                q(i, t+1) = 1;
            else
                q(i, t+1) = 2;
            end
        end
        if q(i, t) == 2 
            if rn > r20(i)
                q(i, t+1) = 2;
            else
                q(i, t+1) = 0;
            end
        end
    end
end   

% initalize cluster means and covariances
Cv      = R * eye(L);      % observation covariance
Cu      = Q * eye(L);      % system's covariance  
u1(1,:) = 2.5 * ones(1,L); % neuron cluster center
u2(1,:) = zeros(1,L);      % noise cluster center

% generate observations
for t = 2 : T
    e = mvnrnd(zeros(1,L), Cu, 1);
    u1(t, :) = u1(t-1, :) + e(1, :);       % kalman process for the neuron cluster
    e = mvnrnd(zeros(1,L), Cu, 1);
    u2(t, :) = u2(t-1, :) + e(1, :);       % stationary noise cluster 
    
    r = rand;
    
    if (q(1, t) == 0 && q(2, t) == 0) 
        spkt(t) = 1;
        obs_id(t) = 1;
        % toss a coin
        rn = rand;
        if rn < 0.5
            cluster_id(t) = 1;
            V(t, :) = mvnrnd(u1(t, :), Cv, 1); % take a spike from the neuron 1
        else 
            cluster_id(t) = 2;
            V(t, :) = mvnrnd(u2(t, :), Cv, 1); % take a spike from the neuron 2
        end
    end
    if (q(1, t) == 0 && q(2, t) ~= 0)
        spkt(t) = 1;
        obs_id(t) = 1;
        cluster_id(t) = 1;
        V(t, :) = mvnrnd(u1(t, :), Cv, 1); % take a spike from the neuron 1
    end
    if (q(1, t) ~= 0 && q(2, t) == 0)
        spkt(t) = 1;
        obs_id(t) = 1;
        cluster_id(t) = 2;
        V(t, :) = mvnrnd(u2(t, :), Cv, 1); % take a spike from the neuron 2
    end
    if (q(1, t) ~= 0 && q(2, t) ~= 0)
        % no observation at this time ...
        obs_id(t) = -1;
        cluster_id(t) = -1;
        V(t, :) = no_obs;
    end       
end

save SynDat_2n3pc.mat Cv Cu Q R V T cluster_id obs_id q r01 r12 r20 u1 u2 spkt;

figure;
for i = 1 : length(V)
    if (cluster_id(i) == 1) ~= 0
        plot3(V(i, 1),V(i, 2),V(i, 3),'.b');   
        hold on;
    end
    if (cluster_id(i) == 2) ~= 0
        plot3(V(i, 1),V(i, 2),V(i, 3),'.r');   
        hold on;
    end
end