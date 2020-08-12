function gradProj = grad_projection(grad1, grad2, d)

% the compoment of g1 that is perpendicular to g2  
     g1 = grad1(:);
     g2 = grad2(:);

     g2 = g2/norm(g2, 2);
     gtemp = g1 - (g2'*g1)*g2;
     gtemp = gtemp/norm(gtemp, 2);   % normalize
gradProj = reshape(gtemp, d, d);
