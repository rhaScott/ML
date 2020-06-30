function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha


m = length(y); % number of training examples

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % a single gradient step on the parameter vector theta
    
    hypothesis = X * theta;
    error = hypothesis - y;
    
    delta = alpha * (X' * error);
    theta = theta - (delta/m);


    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
