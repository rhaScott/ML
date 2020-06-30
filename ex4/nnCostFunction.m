function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% X = (5000x400)
% theta1 = (25x401)
% theta2 = (10*25)




% ////////////////////////////// PART 1 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

%forward propagation of NN

a1 = [ones(m,1) X]; % (5000x401)

z2 = a1 * Theta1'; % (5000x25)
a2 = sigmoid(z2); % (5000x25)
a2 = [ones(size(a2,1),1) a2]; %(5000x26)

z3 = a2 * Theta2'; %(5000x10)
a3 = sigmoid(z3); %(5000x10)

%calculation of cost function J for NN

y_Vec = (1:num_labels)==y; %(5000x10) logical array

pos = -y_Vec .* log(a3);
neg = (1-y_Vec) .* log(1-a3);

J = (1/m) * sum(sum(pos-neg,2));




% ////////////////////////////// PART 2 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

% unvectorised backpropagation algorithm
% loop through all m training data
for t = 1:m 
    
    %%%%% 1) Forward propagate for each m (t)
    %select row t from X and return it as a column vector
    a_1 = X(t,:)'; % n x 1
    a_1 = [1; a_1]; %n+1 x 1
    
    % (hid_layer_size x n+1)(n+1 x1) = (hid_layer_size x 1)
    z_2 = Theta1 * a_1; 
    
    a_2 = sigmoid(z_2); % (hid_layer_size x 1)
    a_2 = [1; a_2]; % (hid_layer_size+1 x 1)
    
    % (K x hid_layer_size+1)(hid_layer_size+1 x 1) = (K x 1)
    z_3 = Theta2 * a_2; 
   
    a_3 = sigmoid(z_3); %(K x 1)
    
    %%%%% 2) Calc delta for each output unit, K
    
    % return logical column vector for t with true when row == y.
    y_Vector = (1:num_labels)' == y(t); % (K x 1)

    % calculate error of classification
    d_3 = a_3 - y_Vector; % (K x 1)
    
    
    %%%%% 3) Backpropagate errors to hidden layer
    % (hid_lay_size+1 x K)(K x 1) = (hid_lay_size+1 x 1)
    d_2 = (Theta2' * d_3) .* [1; sigmoidGradient(z_2)];
    % remove bias unit
    d_2 = d_2(2:end);
    
    %%%%% 4) Update capital theta values & accumulate gradient errors
    Theta1_grad = Theta1_grad + (d_2 * a_1'); 
    Theta2_grad = Theta2_grad + (d_3 * a_2');

end

%%%%% 5) Use accumulated grads to calc (unregularized) gradient for NN cost functions
Theta1_grad = (1/m) * Theta1_grad; % 25 x 401
Theta2_grad = (1/m) * Theta2_grad; % 10 x 26





% ////////////////////////////// PART 3 \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

%%%%% Regularisation of cost functiom (J)

% Calc reg_term for cost function
% note: bias column is not regularised
J_reg_term =  (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Updating cost function to include reg_term
J = J + J_reg_term;

%%%%% Regularisation of Theta_grads

% Calc reg_term for Theta gradients.
Theta1_grad_reg_term = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad_reg_term = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

% Updating Theta_grads to include regularisation terms.

Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
