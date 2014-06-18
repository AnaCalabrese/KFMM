%% simulate data

% define parameters
L = 2;
T = 1000;
J = 4;
R = 2;           % observation noise
Q = 0.008;       % process noise      

% initalize cluster means and covariances
Cv = R * eye(L);        % observation covariance
Cu = Q * eye(L);        % system's covariance  
u1(1,:) = zeros(1,L);   
u2(1,:) = 5 * ones(1,L); 
u3(1,:) = [-5 -5]; 
u4(1,:) =  [-5 5]; 


% first observation
V(1, :) = mvnrnd(u1(1, :), Cv,1);
cluster_id(1) = 1;

for i = 1 : T-1
    e1 = mvnrnd(zeros(1,L), Cu, 1);
    e2 = mvnrnd(zeros(1,L), Cu, 1);
    e3 = mvnrnd(zeros(1,L), Cu, 1);
    e4 = mvnrnd(zeros(1,L), Cu, 1);
    u1(i + 1, :) = u1(i, :) + e1;
    u2(i + 1, :) = u2(i, :) + e2;
    u3(i + 1, :) = u3(i, :) + e1;
    u4(i + 1, :) = u4(i, :) + e2;
    
    r = rand;
    if r <= 0.25
        cluster_id(i+1) = 1;
        V(i + 1, :) = mvnrnd(u1(i+1, :), Cv, 1);
    end
    if (r > 0.25  && r <= 0.5)
        cluster_id(i+1) = 2;
        V(i+1, :) = mvnrnd(u2(i+1, :),Cv, 1);
    end
    if (r > 0.5  && r <= 0.75)
        cluster_id(i+1) = 3;
        V(i+1, :) = mvnrnd(u3(i+1, :),Cv, 1);
    end
    if (r > 0.75  && r <= 1)
        cluster_id(i+1) = 4;
        V(i+1, :) = mvnrnd(u4(i+1, :),Cv, 1);
    end
end

figure;
% plot means trajectories
subplot(1,2,1)
for i = 1 : T
    plot(u1(:,1),u1(:,2),'-k');
    hold on;
    plot(u2(:,1),u2(:,2),'-k');
end 
 
% plot observed data
subplot(1,2,2);
for k = 1 : T
    if (cluster_id(k) == 1) ~= 0
        plot(V(k,1),V(k,2),'.b');   
        hold on;
    end
    if (cluster_id(k) == 2) ~= 0
        plot(V(k,1),V(k,2),'.r');   
        hold on;
    end
    if (cluster_id(k) == 3) ~= 0
        plot(V(k,1),V(k,2),'.g');   
        hold on;
    end
    if (cluster_id(k) == 4) ~= 0
        plot(V(k,1),V(k,2),'.k');   
        hold on;
    end
end

