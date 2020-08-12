function fd_1st_d = fD1(X, D, A, N, d)

% ---------------------------------------------------------------------------
% the gradient of the dissimilarity constraint function w.r.t. A
%
% for example, let distance by L1 norm:
% f = f(\sum_{ij \in D} \sqrt{(x_i-x_j)A(x_i-x_j)'})
% df/dA_{kl} = f'* d(\sum_{ij \in D} \sqrt{(x_i-x_j)^k*(x_i-x_j)^l})/dA_{kl}
%
% note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
% so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij
%     df/dA = f'(\sum_{ij \in D} \sqrt{tr(d_ij'*d_ij*A)})
%             * 0.5*(\sum_{ij \in D} (1/sqrt{tr(d_ij'*d_ij*A)})*(d_ij'*d_ij))
% ---------------------------------------------------------------------------

sum_dist = 0.000001; sum_deri =  zeros(d,d);
%%the  code working directly for tuple pairs is bit slow, so above
%    we converted pair constraints to square adjenecy matrix and use old method
% this may create large matrix if N is large
% if size(D,2)==2
%     D1=sparse(zeros(N,N));
%     D1(D(:, 1), D(:, 2))=1;
%     D=D1;
%     clear D1;
%     for i = 1:N
%         for j= 1:N     % count each pair once
%             if D(i,j) == 1
%                 d_ij = X(i,:) - X(j,:);
%                 [dist_ij, deri_d_ij] = distance1(A, d_ij);
%                 sum_dist = sum_dist +  dist_ij;
%                 sum_deri = sum_deri + deri_d_ij;
%             end
%         end
%     end
%     %sum_dist
%     fd_1st_d = dgF2(sum_dist)*sum_deri;
% end
%     
    %%following code works directly for tuple pairs but slow, so above
    %    we converted pair constraints to square adjenecy matrix and use old method
    %%use following code big data
    
    neg_diff= X(D(:,1),:)-X(D(:,2),:);
    
    for i=1:size(D,1)
        M(:,:,i)=einsum(neg_diff(i,:),neg_diff(i,:),1,1);
    end
    dist = sqrt(einsum(M, A,'ijk,ij->k'));
    tmp= 2*(dist + 1e-6);
    for k=1:length(tmp)
        M(:,:,k)= M(:,:,k)./tmp(k);
    end
    
    sum_deri= sum(M,3);
    fd_1st_d= sum_deri / (sum(dist) + 1e-6);

end
% ------------------------------------------------


% ___________derivative of cover function 1_________
function z = dgF1(y)
z = 1;
end
% ___________derivative of cover function 2_________
function z = dgF2(y)
z = 1/y;
end


function [dist_ij, deri_d_ij] = distance1(A, d_ij)
% distance and derivative of distance using distance1: distance(d) = L1
fudge = 0.000001;  % regularizes derivates a little

M_ij = d_ij'*d_ij;
dist_ij = sqrt(trace(M_ij*A));

% derivative of dist_ij w.r.t. A
deri_d_ij = 0.5*M_ij/(dist_ij+fudge);
end

function [dist_ij, deri_d_ij] = distance2(A, d_ij)
% distance and derivative of distance using distance2: distance(d) = sqrt(L1)
fudge = 0.000001;  % regularizes derivates a little

M_ij = d_ij'*d_ij;
L2 = trace(M_ij*A);           % L2 norm
dist_ij = sqrt(sqrt(L2));

% derivative of dist_ij w.r.t. A
deri_d_ij = 0.25*M_ij/(L2^(3/4)+fudge);
end

function [dist_ij, deri_d_ij] = distance3(A, d_ij)
% distance and derivative of distance using distance3: 1-exp(-\beta*L1)
fudge = 0.000001;  % regularizes derivates a little

beta = 0.5;
M_ij = d_ij'*d_ij;
L1 = sqrt(trace(M_ij*A));
dist_ij = 1 - exp(-beta*L1);

% derivative of dist_ij w.r.t. A
deri_d_ij = 0.5*beta*exp(-beta*L1)*M_ij/(L1+fudge);
end
