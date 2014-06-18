function [P, p, cl_id] = EMMoG4n(DataSet, EMit)
% This functions assigns cluster ids to the simulated data 'dataset' using a 
% mixture of Gaussians model. 

    % Load data set

    load(DataSet);

    J = 4;                      % define number of cluster
    T = length(V(:,1));         % experiment length
    D = length(V(1,:));         % dimensionality of data
    K = 1;                      % scaling factor cov noise cluster
    Tobs = length(V(:,1));
    
    % Parameter Initialization: use simple clustering method (k-means) to
    % determine the centers u1,...,uJ of J components. Set a1,..,aJ = 1/J;
    % Cv_1,...,Cv_J = eye(D). 
    
    % use only the times that have an observation for the kmeans algorithm
    i = 0;
    for t = 1 : length(V)
        i = i+1;
        Vobs(i, :) = V(t, :);          
    end
    
    [IDV, U] = kmeans(Vobs, J);
    [dummy, maxidx] = max(U(:,1));
    
    % initial guesses for neuron cluster
    for j = 1 : J
        P(j).a = 1/J;            % clusters' probabilities
        P(j).u = U(j,:);    % clusters means
        P(j).Cv = eye(D);
    end
    
    % auxiliary structures
    w = zeros(J, T);
    p = zeros(J, T);

    for idx = 1 : EMit
        
        % STEP 2: compute a first guess of pj(i) = p(z(i)=j|guessed parameters), where z 
        % is the identity of the ith waveform and j represents a particular cluster. Then we 
        % have p(j,i) = p(z(i)=j|guessed parameters).
        for i = 1 : T
            normalization = 0;
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


        % STEP 3: with these conditional probabilities update the values of the parameters    
        for j = 1 : J
            % update the cluster probabilities
            P(j).a = sum(p(j, :)) / Tobs;
        end
        for j = 1 : J            
            % compute the new covariance
            Cv_new = zeros(size(P(j).Cv));
            for i = 1 : T
                Cv_new = Cv_new + p(j, i) * (V(i, :) - P(j).u)' * (V(i, :) - P(j).u);
            end   
            Cv_new = Cv_new / sum(p(j, :));
        end
        for j = 1 : J
            % update the means of neurons clusters
            u_new = zeros(size(P(j).u));        
            for i = 1 : T
                u_new = u_new + p(j, i) * V(i, :); 
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
        [dummy, I] = max(p(:,k));
        for j = 1 : J
            if (I == j) ~= 0 
                cl_id(k) = j;
            end
        end
    end  
        
%     % plot data with estimated ids
%     figure;
%     subplot(1,2,1);
%     cla;    
%     for i = 1 : T
%         if (cl_id(i) == 1) ~= 0
%             plot(V(i,1), V(i,2),'.r');
%             hold on;
%         end
%         if  (cl_id(i) == 2) ~= 0
%             plot(V(i,1), V(i,2),'.b');
%             hold on;
%         end
%         if (cl_id(i) == 3) ~= 0
%             plot(V(i,1), V(i,2),'.g');
%             hold on;
%         end
%         if  (cl_id(i) == 4) ~= 0
%             plot(V(i,1), V(i,2),'.k');
%             hold on;
%         end
%     end
% 
%     % plot data with true ids
%     subplot(1,2,2);
%     cla;
%     for k = 1 : T
%         if (cluster_id(k) == 1) ~= 0
%             plot(V(k,1), V(k,2),'.r');   
%             hold on;
%         end
%         if (cluster_id(k) == 2) ~= 0
%             plot(V(k,1), V(k,2),'.b');   
%             hold on;
%         end
%          if (cluster_id(k) == 3) ~= 0
%             plot(V(k,1), V(k,2),'.g');   
%             hold on;
%         end
%         if (cluster_id(k) == 4) ~= 0
%             plot(V(k,1), V(k,2),'.k');   
%             hold on;
%         end
%     end     
end