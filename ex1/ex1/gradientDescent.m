function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

% a single gradient step on the parameter vector theta

    hypothesis = X * theta;
    error = hypothesis - y;
    delta = (1/m) * (X' * error);
    theta = theta - alpha * delta;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
