function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% X         = m x n
% theta     = n x 1
% hyp / z   = m * 1
% y         = m * 1
% pos / neg = m * 1 
% J         = 1 * 1
% grad      = (n * m)(m * 1) = n * 1

% Evaluating the hypothesis for given values of X
z = X * theta;
hyp = 1./(1 + exp(-z));

% Evaluating the positive and negative classes
pos = -y .* log(hyp);
neg = (1-y) .* log(1-hyp);

% Regularisation term for cost function. theta(2:end) as j = 0 (intercept) is not regularised
reg_term = (lambda/ (2*m)) * sum(theta(2:end).^2);

% Evaluating the cost function 
J = (1/m) * sum(pos-neg) + reg_term;

% Evaluating the gradients, treating j = 0 (intercept) as a seperate case as it is not regularised.
grad(1) = (1/m) * (X(:,1)'*(hyp-y));
grad(2:end) = (1/m) * (X(:,2:end)'*(hyp-y)) + (lambda/m) * theta(2:end);

% =============================================================

grad = grad(:);

end
