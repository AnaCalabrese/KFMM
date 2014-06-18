% Load data set
DataSet = 'SynDat4n_v3.mat';
load(DataSet);

Y = V;
clear V;

% Define parameters 
J           = 4;                     % number of clusters
T           = length(Y(:,1));        % experiment length
ss          = 2;                     % state size
em_iters    = 20;                    % number of Em iterations for the GMM
EM_iters    = 20;                    % number of EM iterations for the KFMM

% Structure to store the results
P = struct([]);

%------------------------ INITIALIZATION ---------------------------------%

% Parameter initialization: use GMMto make a first guess of the
% cluster ids. All the true parameters are given to the algorithm.
disp('Initializing : runnign GMM ...');

[initP, initp, init_clid] = EMMoG4n(DataSet, em_iters);

for j = 1 : J
    P(j).Q = Cu;                    % system covariance
    for t = 1 : T
        P(j).x(:,t) = initP(j).u;
        P(j).V(:, :, t) = initP(j).Cv;  % initial guess for the state covariance  
        P(j).R = initP(j).Cv;           % observation covariance 
        P(j).a = 1/J;                   % clusters weights
    end
end

% Auxiliary structures
for j = 1 : J
    P(j).xs  = zeros(ss, T);         % smoothed positions
    P(j).Vs  = zeros(ss, ss, T);     % smoothed covariances
end

% initial guess for the responsibilities
p     = initp;                       % probability that the t-th observation belongs to the j-th cluster. 
                                     % Initially we guess these using the MoG implementation 
                                     

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EM RECURSION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conv_KF = [];

for iter = 1 : EM_iters
    
    % Assign cluster ids
    cl_id = zeros(size(cluster_id));
    for k = 1 : T
        [dummy, I] = max(p(:,k));
        for j = 1 : J
            if (I == j) ~= 0 
                cl_id(k) = j;
            end
        end
    end
    
    cl_idKF = cl_id;

    % Forward step for updating the means
    for j = 1 : J
        for t = 2 : T
            P(j).V(:, :, t) = inv(inv(P(j).V(:, :, t - 1) + P(j).Q) + (p(j, t - 1) * inv(P(j).R)));
            P(j).x(:, t) = P(j).V(:, :, t) * (inv(P(j).V(:, :, t - 1) ...
                + P(j).Q) * P(j).x(:, t - 1) + p(j, t - 1) * (inv(P(j).R) * Y(t - 1, :)'));
        end
    end

    % Backward step for updating the means  
    for j = 1 : J
        P(j).xs(:, T) = P(j).x(:, T);
        P(j).Vs(:, :, T) = P(j).V(:, :, T);
        for t = T - 1 : -1 : 1
            K = P(j).V(:, :, t) * inv(P(j).V(:, :, t) + P(j).Q);                
            P(j).xs(:, t) = P(j).x(:, t) + K * (P(j).xs(:, t + 1) - P(j).x(:, t));
            P(j).Vs(:, :, t) = P(j).V(:, :, t) + K * (P(j).Vs(:, :, t + 1) - (P(j).V(:, :, t) + P(j).Q)) * K';
        end            
        for t = T  : -1 : 1
            P(j).x(:, t) = P(j).xs(:, t);
            P(j).V(:, :, t) = P(j).Vs(:, :, t);
        end
    end
    P(2).xs = P(2).x;

    % Update observation covariance
    for j = 1 : J
        P_new(j).R = 0 * P(j).R;
        for t = 1 : T
            P_new(j).R = P_new(j).R + p(j, t) * (Y(t, :)' - P(j).x(:, t)) * (Y(t, :) - P(j).x(:, t)');
        end   
        P_new(j).R = P_new(j).R / sum(p(j, 1 : T));
    end
    for j = 1 : J            
        P(j).R = P_new(j).R;
    end

    % Update state covariance
    for j = 1 : J
        for t = 1 : T
            P(j).V(:, :, t) = P(j).R;       
        end
    end

    % Estimate probabilities
    for t = 1 : T
        normalization = 0;
         for j = 1 : J
            p(j, t) = exp(- 0.5 * (log(det(P(j).R + P(j).Q)) + (Y(t, :) ...
                - P(j).x(:, t)') * inv(P(j).R + P(j).Q) * (Y(t, :)' - P(j).x(:, t))));
            normalization = normalization + p(j, t);
         end
         p(:, t) = p(:, t) / normalization; 
    end    

end

%%-------------------- PLOT RESULTS -------------------%

figure;
% plot data with ids as given by initiL guess of p    
subplot(1,3,1);
for i = 1 : T
    if (init_clid(i) == 1) ~=0
        plot(Y(i,1), Y(i,2),'.r');
        hold on;
    end
    if  (init_clid(i) == 2) ~=0
        plot(Y(i,1), Y(i,2),'.b');
        hold on;
    end
    if (init_clid(i) == 3) ~=0
        plot(Y(i,1), Y(i,2),'.g');
        hold on;
    end
    if  (init_clid(i) == 4) ~=0
        plot(Y(i,1), Y(i,2),'.k');
        hold on;
    end
end
title('estimated ids');

% plot data with estimated ids
subplot(1,3,2);    
for i = 1 : T
    if (cl_id(i) == 1) ~=0
        plot(Y(i,1), Y(i,2),'.r');
        hold on;
    end
    if  (cl_id(i) == 2) ~=0
        plot(Y(i,1), Y(i,2),'.b');
        hold on;
    end
    if (cl_id(i) == 3) ~=0
        plot(Y(i,1), Y(i,2),'.g');
        hold on;
    end
    if  (cl_id(i) == 4) ~=0
        plot(Y(i,1), Y(i,2),'.k');
        hold on;
    end
end
title('estimated ids');

% % plot data with true ids
subplot(1,3,3);
for k = 1 : T
    if (cluster_id(k) == 1) ~= 0
        plot(Y(k,1), Y(k,2),'.b');   
        hold on;
    end
    if (cluster_id(k) == 2) ~= 0
        plot(Y(k,1), Y(k,2),'.g');   
        hold on;
    end
     if (cluster_id(k) == 3) ~= 0
        plot(Y(k,1), Y(k,2),'.k');   
        hold on;
    end
    if (cluster_id(k) == 4) ~= 0
        plot(Y(k,1), Y(k,2),'.r');   
        hold on;
    end
end
title('true ids');

