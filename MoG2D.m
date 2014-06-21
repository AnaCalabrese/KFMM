function [P, p, cl_id] = MoG2D(DataSet, J, EMit)
% This functions assigns cluster ids to the simulated data 'DataSet' using a 
% mixture of Gaussians model. 

    % load data
    load(DataSet);
    
    
    T = length(V(:,1));         % experiment length
    D = length(V(1,:));         % dimensionality of data
    K = 1;                      % scaling factor cov noise cluster
    Tobs = sum(spkt);
    
    
    % Parameter Initialization: use simple a k-means clustering algorithm to
    % determine the centers u1,...,uJ of J components. Set a1,..,aJ = 1/J;
    % Cv_1,...,Cv_J = eye(D). 
    
    % Use only the times that have an observation for the kmeans algorithm
    i = 0; Nobs = sum(obs_id==1); Vobs = zeros(Nobs, D);
    for t = 1 : T
        if obs_id(t) == 1
            i = i+1;
            Vobs(i, :) = V(t, :);          
        end
    end
    
    [~, U] = kmeans(Vobs, J);
    
    % initial guesses for neuron cluster
    P = struct([]);   
    for j = 1 : J
        P(j).a = 1/J;       % clusters' probabilities
        P(1).u = max(U);    % clusters means
        P(2).u = min(U);    % clusters means
        P(j).Cv = eye(D);   % observation noise
    end
    
    % auxiliary structures
    w = zeros(J, T);
    p = zeros(J, T);   
    
    for idx = 1 : EMit
        
        % Compute a first guess of pj(i) = p(z(i)=j|guessed parameters), where z 
        % is the identity of the ith waveform and j represents a particular cluster. Then we 
        % have p(j,i) = p(z(i)=j|guessed parameters).
        for i = 1 : T
            if obs_id(i) == 1
                normalization = 0;
                mm = zeros(1,J);
                for j = 1 : J
                    mm(j) = log(P(j).a) - 0.5 * (log(det(P(j).Cv)) + (V(i, :) - P(j).u) * inv(P(j).Cv) * (V(i, :) - P(j).u)');
                end
                    m = max(mm);
                for j = 1 : J
                    w(j, i) = exp(log(P(j).a) - 0.5 * (log(det(P(j).Cv)) + (V(i, :) - P(j).u) * inv(P(j).Cv) * (V(i, :) - P(j).u)') - m);
                    normalization = normalization + w(j, i);
                end
                p(:, i) = w(:, i) / normalization;
            end
            if obs_id(i) == -1
                p(:, t) = [0, 0];
            end
        end


        % With these conditional probabilities update the values of the parameters    
        for j = 1 : J
            % update the cluster probabilities
            P(j).a = sum(p(j, :)) / Tobs;
        end
        for j = 1 : J            
            % compute the new covariance
            Cv_new = zeros(size(P(j).Cv));
            for i = 1 : T
                if obs_id(i) == 1
                    Cv_new = Cv_new + p(j, i) * (V(i, :) - P(j).u)' * (V(i, :) - P(j).u);
                end
            end   
            Cv_new = Cv_new / sum(p(j, :));
        end
        for j = 1 : J  
            % update the means of neurons clusters
            u_new = zeros(size(P(j).u));        
            for i = 1 : T
                if obs_id(i) == 1
                    u_new = u_new + p(j, i) * V(i, :); 
                end
            end 
            u_new = u_new / sum(p(j, :));
            P(j).u = u_new;
        end
        for j = 1 : J
            % update the covariance
            P(j).Cv = Cv_new;
        end    
     end
    
    % assign cluster ids
    cl_id = zeros(size(cluster_id));
    for k = 1 : T
        if obs_id(k) == 1
            [~, I] = max(p(:,k));
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
end