function fd = fD(X, D, A, N)

% ---------------------------------------------------------------------------
% the value of dissimilarity constraint function
% f = f(\sum_{ij \in D} distance(x_i, x_j)) 
% i.e. distance can be L1:  \sqrt{(x_i-x_j)A(x_i-x_j)'}) ...
%      f(x) = x ...
% ---------------------------------------------------------------------------

if size(D,1)==size(D,2)
fd = 0.000001;

for i = 1:N
  for j= 1:N
    if D(i,j) == 1
      d_ij = X(i,:) - X(j,:);
      distij = distance1(A, d_ij);      % distance between 'i' and 'j'
      fd = fd + distij;        % constraint defined on disimilar set
    end   
  end
end

else

neg_diff= X(D(:,1),:)-X(D(:,2),:);
fd= log(sum(sqrt(sum((neg_diff*A).*neg_diff, 2))));

fd = gF2(fd);
end
end

% ___________L1 norm______________
function kd = distance1(A, d_ij)
kd = (d_ij * A * d_ij')^(1/2);
end
% ___________sqrt(L1 norm)___________
function kd = distance2(A, d_ij)
kd = (d_ij * A * d_ij')^(1/4);
end
% ___________1-exp(-beta*L1)_________
function kd = distance3(A, d_ij)
beta = 0.5;
kd = 1 - exp(-beta*(sqrt(d_ij * A * d_ij')));
end
% ___________cover function 1_________
function x = gF1(x1)
x = x1;  
end
% ___________cover function 1_________
function x = gF2(x1)
x = log(x1);  
end


