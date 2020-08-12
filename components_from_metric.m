function L = components_from_metric(metric)
% %   """Returns the transformation matrix from the Mahalanobis matrix.
% %   Returns the transformation matrix from the Mahalanobis matrix, i.e. the
% %   matrix L such that metric=L.T.dot(L).
% %   Parameters
% %   ----------
% %   metric : symmetric `np.ndarray`, shape=(d x d)
% %     The input metric, from which we want to extract a transformation matrix.
% %   tol : positive `float`, optional
% %     Eigenvalues of `metric` between 0 and - tol are considered zero. If tol is
% %     None, and w_max is `metric`'s largest eigenvalue, and eps is the epsilon
% %     value for datatype of w, then tol is set to w_max * metric.shape[0] * eps.
% %   Returns
% %   -------
% %   L : np.ndarray, shape=(d x d)
% %     The transformation matrix, such that L.T.dot(L) == metric.
% %   """

if sum(sum(metric-metric'))>0.000001
    error('The input metric should be symmetric.')
end
if isdiag(metric)
   L= diag(sqrt(max(0,diag(metric))));
else
   [L, flag] =chol(metric);
   L=L';
   if flag~=0 %not psd   
       warning('input is not positive definite') %should be error
      [V,D] = eig(metric);
       w= diag(D);    
       tol = max(abs(w))*length(w)*eps(16); %eps(max(w));
       if tol<0
           error('tol should be positive');
       elseif any(w<-tol)
           error('Non PSD error');
       else
         L= V'.*sqrt(max(0, w));
       end
   end     
end