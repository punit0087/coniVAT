function [C, ml, cl] = get_mustlink_cannotlink(input_data, labels, ml_percentile, cl_percentile, max_num_constraints)

%%Input Arguments
%%Input data
% Labels = partial labeled data, -1 for unknown labels
% Ml_percent = kernal_similarity width (in %) for ml constraints
% Cl_percent= kernal_disimilarity width (in %) for cl constraints
% max_num_constraints- number of constraints queried 

[full_data_labled_idx]= find(labels~=-1);

labled_data_labels= labels(labels~=-1);
classes= unique(labled_data_labels);

C = nchoosek(1:length(labled_data_labels),2);
C=[C -1*ones(size(C,1),1)];
for i=1:length(classes)
    this_clus= find(labled_data_labels==classes(i));
    C1 = nchoosek(this_clus,2);
    [~,ia,~]= intersect(C(:,1:2),C1,'rows');
    C(ia,3)= 1;
end

rs= pdist(input_data(full_data_labled_idx,:));
rs= squareform(rs);

idx=sub2ind(size(rs),C(:,1),C(:,2));
dist=rs(idx);
C(:,4)=dist;
%replace data points index with respect to full data points index
C(:,1) = changem(C(:,1),full_data_labled_idx, 1:length(labled_data_labels));
C(:,2) = changem(C(:,2),full_data_labled_idx, 1:length(labled_data_labels));



ml_threshold= prctile(C(C(:,3)==1,4),ml_percentile);
cl_threshold= prctile(C(C(:,3)==-1,4),cl_percentile);

C(C(:,3)==1 & C(:,4)< ml_threshold,5) = 1;
C(C(:,3)==-1 & C(:,4)> cl_threshold ,5) = 1;

ml = C(C(:,3)==1 & C(:,5)==1,1:2);
cl = C(C(:,3)==-1 & C(:,5)==1,1:2);

total_ml_constraints= min(size(ml,1), max_num_constraints);
total_cl_constraints= min(size(cl,1), max_num_constraints);

ml= ml(randperm(size(ml,1),total_ml_constraints),:);
cl= cl(randperm(size(cl,1),total_cl_constraints),:);
end