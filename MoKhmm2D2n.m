function MoKhmm2D2n
    clear all;

    % Load data set
    DataSet = 'SynDat3s_2n_v1.mat';
    load(DataSet);
    Y = V;
    clear V;
    
    % Define parameters and auxiliary structures
    EM_iters = 10;               % number of EM iterations       
    ss       = 2;                % number of cells
    J        = 2;                % number of clusters
    T        = length(Y(:,1));   % experiment length
    Col      = 9;                % number of sates in the joint cluster model (NxN, with N = 3)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    % Structure to store the results
    P = struct([]);

    % Parameter initialization: use MoG model to make a first guess of the
    % cluster ids. 
    disp('Initialization: running MoG ...')
    [GMMP, GMMp, cl_id] = MoG2D(DataSet, 20);
    
    for j = 1 : J
        P(j).R = GMMP(j).Cv;                % observation covariance
        P(j).Q = Cu;                        % system covariance
        for t = 1 : T
            P(j).x(:,t) = GMMP(j).u;        % cluster positions
            P(j).V(:, :, t) = GMMP(j).Cv;   % initial guess for the state covariance
        end
    end
   
    % Auxiliary structures
    for j = 1 : J
        P(j).xs  = zeros(ss, T);         % smoothed positions
        P(j).Vs  = zeros(ss, ss, T);     % smoothed covariances
    end

    % Initial guess for the responsibilities
    p  = GMMp;          % probability that the t-th observation belongs to the j-th cluster. 
                        % Initially we guess these using the GMM 
    pq = zeros(T, Col); % p(q^t|V), dim: (T, S)      
    States  = q;        % true states of neuron cluster 
    clear q;
    
    % Fixed parameters for HMM
    r01 = 1;                 % transition rate from spike state to r.p. state
    r12 = 0.5;               % transition rate from r.p. state to isi state
    
    % Estimate mean isi for each neuron 
    [isi_1 isi_2] = estim_isi(cl_id);
    r_isi1 = 1/isi_1;
    r_isi2 = 1/isi_2;
   
    % Define transition and observation matrices for HMM 
    a1 = [1-r01 r01 0; 0 1-r12 r12; r_isi1 0 1-r_isi1]';  % the transition matrix for the HMM 1
    a2 = [1-r01 r01 0; 0 1-r12 r12; r_isi2 0 1-r_isi2]';  % the transition matrix for the HMM 2
    Pa = kron(a1,a2);                                     % joint transition matrix
    Pb = [0.5 0.5 0 ; 1 0 0 ; 1 0 0 ; 0 1 0 ; 0 0 1 ; ...
        0 0 1 ; 0 1 0 ; 0 0 1 ; 0 0 1];                   % joint observation matrix p(z|q1q2)
    
    % Initialize Forward step for p(q^t|V)
    % initial probability distribution: p(q^1 = i).
    % at t = 1, the system is in state i = 9 (q1=3,q2=3).
    Pi   = [0 0 0 0 0 0 0 0 1];       

    % Initialize forward distribution alpha
    alpha = zeros(T, Col); 
    ti = 1; 
    for i = 1 : Col
        biV = 0;
        if obs_id(ti) == 1         
            for j = 1 : J
                biV = biV + (exp(-0.5*(log(det(P(j).R + P(j).Q)) + (Y(ti, :) ...
                - P(j).x(:, ti)') * ((P(j).R + P(j).Q) \ (Y(ti, :)' - P(j).x(:, ti))))) * Pb(i, j));
            end
        else
            biV = biV + Pb(i, 3);
        end
        alpha(ti, i) = Pi(i) * biV;
    end
     
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Start EM recursion
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Running MoKhmm ...');
    
    for iters = 1 : EM_iters
                
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
               
        % assign state ids
        StateIdxs = zeros(1, T);
        for t = ti : T
            [dummy I] = max(pq(t,:));
            StateIdxs(t) = I;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % M-step
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % E-step
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Induction for alpha (p(V^1:t, q^t = i | params))
        for t = ti+1 :  T
            norm_alpha = 0;
            for k = 1 : Col
                temp = 0;
                for i = 1 : Col
                    temp = temp + alpha(t-1, i) * Pa(k, i);
                end
                bkV = 0;
                if obs_id(t) == -1
                    bkV = Pb(k, 3);
                else
                    for j = 1 : J
                        bkV = bkV + (exp(-0.5*(log(det(P(j).R + P(j).Q)) + (Y(t, :) ...
                        - P(j).xs(:, t)') * ((P(j).R + P(j).Q) \ (Y(t, :)' - P(j).xs(:, t))))) * Pb(k, j));
                    end
                end 
                alpha(t, k) = temp * bkV;
                norm_alpha = norm_alpha + alpha(t, k);
            end
            alpha(t, :) = alpha(t, :) / norm_alpha;
        end

        % Initialize backward step
        beta       = zeros(T, Col); 
        beta(T, :) = ones(1, Col);

        % Induction for beta (p(V^t+1:T|q^t = i))
        for t = T-1 : -1 : ti
            norm_beta = 0;
            if obs_id(t) == 1
                for i = 1 : Col
                    temp = 0;
                    for k = 1 : Col
                        bkV = 0;
                        for j = 1 : J
                            bkV = bkV + (exp(-0.5*(log(det(P(j).R + P(j).Q)) + (Y(t, :) ...
                            - P(j).xs(:, t)') * ((P(j).R + P(j).Q) \ (Y(t, :)' - P(j).xs(:, t))))) * Pb(k, j));
                        end
                        temp = temp + Pb(k, 3) * Pa(k, i) * beta(t+1, k);
                    end
                    beta(t, i) = temp;
                    norm_beta = norm_beta + temp;
                end
            else
                for i = 1 : Col
                    temp = 0;
                    for k = 1 : Col 
                        temp = temp + Pb(k, 3) * Pa(k, i) * beta(t+1, k);
                    end
                    beta(t, i) = temp;
                    norm_beta = norm_beta + temp;
                end
            end
            beta(t, :) = beta(t, :) / norm_beta;
        end

        % Forward-backward probabilities
        pq = zeros(T, Col); % p(q^t|V^1:T), dim: (T, S)    
        for t = 1 : T 
            pq_norm = sum(alpha(t, :) * beta(t, :)');
            pq(t,:) = (alpha(t,:) .* beta(t,:))' ./ pq_norm;
        end
        

        % Re-compute responsibilities: p(z^t|V^1:T)
        p = zeros(J, T);  
        obs_t = find(obs_id == 1);
        for k = 1 : length(obs_t)
            t = obs_t(k);
            norm_p  = 0;
            for j = 1 : J
                temp = 0;
                for i = 1 : Col
                    temp = temp + pq(t, i) * Pb(i, j); 
                end 
                p(j, t) = temp * exp(-0.5*(log(det(P(j).R + P(j).Q)) + (Y(t, :) ...
                        - P(j).xs(:, t)') * ((P(j).R + P(j).Q) \ (Y(t, :)' - P(j).xs(:, t))))); 

                norm_p = norm_p + p(j, t);                 
            end
            p(:, t) = p(:, t) / norm_p;
        end
     end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Termination
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Terminating ...');
    
    % Compute error bars for the means estimates
    iV = zeros(J, J, T);
    for j = 1 : J
        for t = 1 : T 
            iV(:, :, t) = inv(P(j).V(:, :, t));
            P(j).eb1(t) = sqrt(inv(iV(1,1,t) - iV(1,2,t) * (1/iV(2,2,t)) * iV(2,1,t))); 
            P(j).eb2(t) = sqrt(inv(iV(2,2,t) - iV(2,1,t) * (1/iV(1,1,t)) * iV(1,2,t)));
        end
    end
        
    % Compute error bars for the MoG 
    for j = 1 : J
        iCv = inv(GMMP(j).Cv);
        GMMP(j).eb1 = sqrt(inv(iCv(1,1) - iCv(1,2) * (1/iCv(2,2)) * iCv(2,1))); 
        GMMP(j).eb2 = sqrt(inv(iCv(2,2) - iCv(2,1) * (1/iCv(1,1)) * iCv(1,2)));    
    end
        
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % Plot results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure;
    % plot data with MoG ids
    subplot(1,3,1);
    for i = 1 : T
        if (GMMp(1, i) > 0.5)
            plot(Y(i,1), Y(i,2),'.r');
            hold on;
        end
        if (GMMp(1, i) < 0.5 && GMMp(2, i) > 0.5) 
            plot(Y(i,1), Y(i,2),'.k');
            hold on;
        end
    end
    title('MoG ids');
    error_ellipse(GMMP(1).Cv, GMMP(1).u);
    error_ellipse(GMMP(2).Cv, GMMP(2).u);
  
    % plot data with MoKhmm ids
    subplot(1,3,2);    
    for i = 1 : T
        if (cl_id(i) == 1) ~=0
            plot(Y(i,1), Y(i,2),'.k','MarkerSize', 10);
            hold on;
        end
        if  (cl_id(i) == 2) ~=0
            plot(Y(i,1), Y(i,2),'.r','MarkerSize', 10);
            hold on;
        end
    end    
    title('MoKhmm ids');
    for t = 1:2000:T
        error_ellipse(P(1).R, P(1).x(:,t));
        error_ellipse(P(2).R, P(2).x(:,t));
    end

    % Plot data with true ids
    subplot(1,3,3);
    for k = 1 : T
        if (cluster_id(k) == 1) ~= 0 
            plot(Y(k,1), Y(k,2),'.k', 'MarkerSize', 10);   
            hold on;
        end
        plot(P(1).x(:,1))
        if (cluster_id(k) == 2) ~= 0
            plot(Y(k,1), Y(k,2),'.r', 'MarkerSize',10);   
            hold on;
        end
    end
    title('true ids');
        
 
    % Mean tracking cell 1
    figure;
    subplot(2,2,1);
    plot(1:T, u1(1:T,1),'-k', 'LineWidth', 2);
    hold on;
    % hmmkf results
    plot(1:T, P(1).x(1,1:T),'-', 'LineWidth', 2);
    errorbar(1:100:T, P(1).x(1,1:100:T), P(1).eb1(1:100:T));
    % gmm results
    u11 = GMMP(1).u(1)*ones(1,T);
    e11 = GMMP(1).eb1*ones(1,T);
    plot(1:T, u11,'-r');
    errorbar(1:500:T, u11(1,1:500:T), e11(1:500:T),'r');    
    xlim([1 T]);
    xlabel('t');
    ylabel('u1_x');
    legend('true', 'MoKhmm');

    subplot(2,2,2);
    plot(1:T, u1(1:T,2),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(1).x(2,1:T),'-', 'LineWidth', 2);
    errorbar(1:100:T, P(1).x(2,1:100:T), P(1).eb2(1:100:T));
    % gmm results
    u12 = GMMP(1).u(2)*ones(1,T);
    e12 = GMMP(1).eb2*ones(1,T);
    plot(1:T, u12,'-r');
    errorbar(1:500:T, u12(1,1:500:T), e12(1:500:T),'r');
    xlim([1 T]);
    xlabel('t');
    ylabel('u1_y');
    
    % Mean tracking cell 2
    subplot(2,2,3);
    plot(1:T, u2(1:T,1),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(2).x(1,1:T),'-', 'LineWidth', 2);
    errorbar(1:100:T, P(2).x(1,1:100:T), P(2).eb1(1:100:T));
    % gmm results
    u21 = GMMP(2).u(1)*ones(1,T);
    e21 = GMMP(2).eb1*ones(1,T);
    plot(1:T, u21,'-r');
    errorbar(1:500:T, u21(1,1:500:T), e21(1:500:T),'r');
    xlim([1 T]);
    xlabel('t');
    ylabel('u2_x');
    
    subplot(2,2,4);
    plot(1:T, u2(1:T,2),'-k', 'LineWidth', 2);
    hold on;
    plot(1:T, P(2).x(2,1:T),'-', 'LineWidth', 2);
    errorbar(1:100:T, P(2).x(2,1:100:T), P(2).eb2(1:100:T));
    % gmm results
    u22 = GMMP(2).u(2)*ones(1,T);
    e22 = GMMP(2).eb2*ones(1,T);
    plot(1:T, u22,'-r');
    errorbar(1:500:T, u22(1,1:500:T), e22(1:500:T),'r');
    xlim([1 T]);
    xlabel('t');
    ylabel('u2_y');
    
end
