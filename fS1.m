function fs_1st_d = fS1(X, S, A, N, d)

% the gradient of the similarity constraint function w.r.t. A
% f = \sum_{ij}(x_i-x_j)A(x_i-x_j)' = \sum_{ij}d_ij*A*d_ij'
% df/dA = d(d_ij*A*d_ij')/dA
%
% note that d_ij*A*d_ij' = tr(d_ij*A*d_ij') = tr(d_ij'*d_ij*A)
% so, d(d_ij*A*d_ij')/dA = d_ij'*d_ij

[N d] = size(X);

if size(S,1)==size(S,2)
    fs_1st_d = zeros(d,d);
    
    fudge = 0.000001;  % regularizes derivates a little if necessary
    
    for i = 1:N
        for j= 1:N
            if S(i,j) == 1
                d_ij = X(i,:) - X(j,:);
                % distij = d_ij * A * d_ij';         % distance between 'i' and 'j'
                % full first derivative of the distance constraints
                fs_1st_d = fs_1st_d + d_ij'*d_ij;
            end
        end
    end
else
    %%
    pos_diff =  X(S(:,1),:)-X(S(:,2),:);
    fs_1st_d=einsum(pos_diff,pos_diff,'ij,ik->jk');
end

end
