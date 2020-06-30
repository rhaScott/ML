function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add a column of ones to X, as intercept values. Assign as a1.
% X = (5000x400) = m * n
% a1 = (5000x401) = m * (n + 1)

a1 = [ones(m,1) X];

% calculate a2 by applying sigmoid function to z2 (linear hypothesis)
% Theta1 = (25x401) 
% where each of the 400 (bar 1st) columns is representative of the mapping 
% of theta values for i-th input layer node to every 2nd layer node.
% As such rows are thisbrelationship but reversed.

% z2 = (5000x401)(401x25) = (5000x25)
% a2 = (5000x25)

z2 = a1 * Theta1';
a2 = sigmoid(z2);

% Concatenate a column of ones to start of a2.
% a2 = (5000x26)
a2 =  [ones(size(a2,1),1) a2];

% calculate a3 by applying sigmoid function to z3 (linear hypothesis)

% z3 = (5000x26)(26x10) = (5000x10)
% a3 = (5000x10) = where each row is the probability of a sample belonging
% to each of the 10 classes. Max of each row finds most likely case.

z3 = a2 * Theta2';
a3 = sigmoid(z3);

[prob,p] = max(a3, [], 2);
% =========================================================================


end
