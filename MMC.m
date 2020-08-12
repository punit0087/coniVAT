function [X_transformed, metric, converge] = MMC(X, S, D)
% Input
% X: data
% S: similarity constraints (in the form of a pairwise-similarity matrix or tuple)
% D: disimilarity constraints (in the form of a pairwise-disimilarity matrix or tuple)

% % 0 if similarity/dissimilarity constraint is in square adjencency matrix form
% % 1 if imilarity/dissimilarity is in pairwise tuple (or list) form
[N,d] = size(X);

% S1= sparse(zeros(N,N));
% D1=sparse(zeros(N,N));
% S1(S(:, 1), S(:, 2))=1;
% D1(D(:, 1), D(:, 2))=1;
% S=S1;
% D=D1;
% clear S1 D1;


A_init = eye(d,d); %  or eye(d,d) *0.1;
maxiter=100;

if size(S,1)==size(S,2) && size(D,1)==size(D,2)
    % number of examples   % dimensionality of examples
    W = zeros(d,d);
    
    for i = 1:N
        for j = 1:N
            if S(i,j) == 1
                d_ij = X(i,:) - X(j,:);
                W = W + (d_ij'*d_ij);
            end
        end
    end
    
    w= W(:);
    t = w' * A_init(:)/100;
    
elseif size(S,2)==2 && size(D,2)==2
    
    pos_diff= X(S(:,1),:)-X(S(:,2),:);
    w=einsum(pos_diff,pos_diff,'ij,ik->jk'); %%or w=einsum(pos_diff,pos_diff,1,1);
    w=w(:);
    
    t = dot(w,A_init(:))/ 100.0;
else
    error('invalid similarity and dissimilarity constraints');
end


[metric, converge] = fit_full_mmc(X, S, D, A_init, w, t, maxiter);
L = components_from_metric(metric);
X_transformed= real(X*L');



