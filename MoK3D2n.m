function MoK3D2n
    % Load data set
    DataSet = 'SynDat_2n3pc.mat';

    load(DataSet);

    Y = V;
    clear V;

    % Define parameters 
    J           = 2;                     % number of clusters
    T           = length(Y(:,1));        % experiment length
    ss          = 3;                     % state size
    em_iters    = 20;                    % number of Em iterations for the GMM
    EM_iters    = 10;                    % number of EM iterations for the KFMM

    % Structure to store the results
    P = struct([]);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Initialization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Parameter initialization: use GMMto make a first guess of the
    % cluster ids. All the true parameters are given to the algorithm.
    disp('Initializing : running MoG ...');

    [initP, initp] = MoG3D(DataSet, em_iters);

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
    disp('running MoK ...');
    conv_KF = [];

    for iter = 1 : EM_iters

        % Assign cluster ids
        cl_id = zeros(size(cluster_id));
        for k = 1 : T
            if obs_id(k) == 1
                [dummy, I] = max(p(:,k));
                for j = 1 : J
                    if (I == j) ~= 0 
                        cl_id(k) = j;
                    end
                end
            end
            if obs_id(k) == -1
                cl_id(k) = -1;
            end
        end

        cl_idKF = cl_id;

        % Forward step for updating the means
        for j = 1 : J
            for t = 2 : T
                % if there is an observation at t-1 ...
                if obs_id(t-1) == 1
                    P(j).V(:, :, t) = inv(inv(P(j).V(:, :, t - 1) + P(j).Q) + (p(j, t - 1) * inv(P(j).R)));
                    P(j).x(:, t) = P(j).V(:, :, t) * (inv(P(j).V(:, :, t - 1) ...
                        + P(j).Q) * P(j).x(:, t - 1) + p(j, t - 1) * (inv(P(j).R) * Y(t - 1, :)'));
                end
                 if obs_id(t-1) == -1
                    % if there is NO observation ...
                    P(j).V(:, :, t) = inv(inv(P(j).V(:, :, t - 1) + P(j).Q));
                    P(j).x(:, t) = P(j).V(:, :, t) * (inv(P(j).V(:, :, t - 1) + P(j).Q) * P(j).x(:, t - 1));
                end
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
            if obs_id(t) == 1
                % if there is an observation at t ...
                normalization = 0;
                 for j = 1 : J
                    p(j, t) = exp(- 0.5 * (log(det(P(j).R + P(j).Q)) + (Y(t, :) ...
                        - P(j).x(:, t)') * inv(P(j).R + P(j).Q) * (Y(t, :)' - P(j).x(:, t))));
                    normalization = normalization + p(j, t);
                 end
                 p(:, t) = p(:, t) / normalization; 
            end
            if obs_id(t) == -1
                % if there is no observation at t, put anything here,
                % just for the sake of having something there, but it
                % is not going to be used.
                p(:, t) = [0, 0];
            end
        end    

    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Termination
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Terminating...');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Plot results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    figure;
    % plot data with ids as given by initial guess of p    
    subplot(1,3,1);
    for i = 1 : T
        if (initp(1, i) > 0.5)
            plot3(Y(i,1), Y(i,2), Y(i,3),'.r');
            hold on;
        end
        if (initp(1, i) < 0.5 && initp(2, i) > 0.5) 
            plot3(Y(i,1), Y(i,2), Y(i,3),'.k');
            hold on;
        end
    end
    title('MoG ids');

    % plot data with estimated ids
    subplot(1,3,2);    
    for i = 1 : T
        if (cl_id(i) == 1) ~=0
            plot3(Y(i,1), Y(i,2), Y(i,3),'.r');
            hold on;
        end
        if  (cl_id(i) == 2) ~=0
            plot3(Y(i,1), Y(i,2), Y(i,3),'.k');
            hold on;
        end
    end
    title('MoK ids');

    % plot data with true ids
    subplot(1,3,3);
    for i = 1 : T
        if (cluster_id(i) == 1) ~= 0
            plot3(Y(i,1), Y(i,2), Y(i,3),'.r');   
            hold on;
        end
        if (cluster_id(i) == 2) ~= 0
            plot3(Y(i,1), Y(i,2), Y(i,3),'.k');   
            hold on;
        end
    end
    title('true ids');

    % plot estimated cluster means together with true cluster means as a
    % function of time
    % cell 1
    figure;
    subplot(2,3,1);
    plot(1:T, u1(1:T,1),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(1).x(1,1:T),'-', 'LineWidth', 2);
    xlim([1 T]);
    xlabel('t');
    ylabel('u1_x');
    legend('true', 'MoK');

    subplot(2,3,2);
    plot(1:T, u1(1:T,2),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(1).x(2,1:T),'-', 'LineWidth', 2);
    xlim([1 T]);
    xlabel('t');
    ylabel('u1_y');
    
    subplot(2,3,3);
    plot(1:T, u1(1:T,3),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(1).x(3,1:T),'-', 'LineWidth', 2);
    xlim([1 T]);
    xlabel('t');
    ylabel('u1_z');

    % cell 2
    subplot(2,3,4);
    plot(1:T, u2(1:T,1),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(2).x(1,1:T),'-', 'LineWidth', 2);
    xlim([1 T]);
    xlabel('t');
    ylabel('u2_x');

    subplot(2,3,5);
    plot(1:T, u2(1:T,2),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(2).x(2,1:T),'-', 'LineWidth', 2);
    xlim([1 T]);
    xlabel('t');
    ylabel('u2_y');
    
    subplot(2,3,6);
    plot(1:T, u2(1:T,3),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(2).x(3,1:T),'-', 'LineWidth', 2);
    xlim([1 T]);
    xlabel('t');
    ylabel('u2_z');
end





