% This function provides an example of how to apply a Kalman filter mixture 
% model to the spike sorting problem presented in A. Calabrese and L.
% Paninski (2010). The example provided corresponds to simulated spike data for
% 2 neurons and a 2-dimensional data representation, although the case of more
% neurons (clusters) and higher dimensional data is analogous. 
% Please see the manuscript included in this directory (KFM.pdf)
% for details about the method and notation.
% Use this code at your own risk but please reference our work.

% This function takes as input the clustering method (MoK or MoKhmm).

function run_model(model)
    % INPUT (examples provided):
    %   - 'MoK' 
    %   - 'MoKhmm'
    
    % OUTPUT: 2 figures, containing clustering results and mean-tracking.
       
    switch model
        case 'MoK'   
            MoK2D;
        case 'MoKhmm'
            % inlcudes a Hidden Markov model to detect refractory period 
            % violations
            MoKhmm2D; 
    end
end