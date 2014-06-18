% This function provides a couple of examples of the performance of a
% Kalman filter mixture model applied to the spike sorting problem, presented 
% in A. Calabrese and L. Paninski (2010). The examples provided are based on 
% synthetic data. Please see the manuscript included in this directory (KFM.pdf)
% for details about the method and notation. Neither this nor the provided 
% code are in any way production code. Use this code at your own risk but 
% reference our work please.

% Executing this functions requires that we clustering method (MoK or 
% MoKhmm), the number of dimensions in the representation of the data 
% (2D or 3D), and the number of cluster in the mixture (2n or 4n). 

function run_model(model)
    % INPUT: clustering method. Examples provided:
    %   - 'MoK2D2n' 
    %   - 'MoKhmm2D2n'
    %   - 'MoK3D2n' 
    %   - 'MoKhmm3D2n'
    %   - 'MoK2D4n'
    
    % OUTPUT: 2 figures, containing clustering results and mean-tracking.
       
    switch model
        case 'MoK2D2n'          
            MoK2D2n;
        case 'MoK3D2n'
            MoK3D2n;
        case 'MoKhmm2D2n'
            MoKhmm2D2n;
        case 'MoKhmm3D2n'
        MoKhmm3D2n;
        case 'MoK2D4n'
        MoK2D4n;
    end
end