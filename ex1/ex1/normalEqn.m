function [theta] = normalEqn(X, y)

%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = (X'*X)^-1*(X'*y)

end
