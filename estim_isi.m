function [est_isi_1 est_isi_2] = estim_isi(cl_id)

    % get estimated spike times for each cluster
    GMM_spkt_1 = find(cl_id == 1);
    GMM_spkt_2 = find(cl_id == 2);

    % get isis for cluster 1
    idx = 0;
    for k = 2 : length(GMM_spkt_1)
        idx = idx+1;
        GMM_isi_1(idx) = GMM_spkt_1(k) - GMM_spkt_1(k-1); 
    end
    
    % get isis for cluster 2
    idx = 0;
    for k = 2 : length(GMM_spkt_2)
        idx = idx+1;
        GMM_isi_2(idx) = GMM_spkt_2(k) - GMM_spkt_2(k-1); 
    end

    est_isi_1 = mean(GMM_isi_1);
    est_isi_2 = mean(GMM_isi_2);
end
