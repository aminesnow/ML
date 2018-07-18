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



a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(m, 1) (sigmoid(z2))];
z3 = a2*Theta2';
h_theta = sigmoid(z3);


y_recoded = zeros(size(h_theta));

for i = 1:numel(y)
  y_recoded(i,y(i)) = 1;
end


J = sum((-(y_recoded.*log(h_theta) + (1 - y_recoded).*log(1 - h_theta)))(:));

J = (J + (lambda/2)*(sum((Theta1(:,2:end).^2)(:)) + sum((Theta2(:,2:end).^2)(:))))/m;


D1_i = zeros(size(Theta1));
D2_i = zeros(size(Theta2));
for i = 1:m
  a1_i = a1(i,:);
  z2_i = z2(i,:);
  a2_i = a2(i,:);
  z3_i = z3(i,:);
  a3_i = h_theta(i,:);
    
  delta3_i = (a3_i - y_recoded(i,:))';
  
  delta2_i = ((Theta2'*delta3_i)(2:end)).*sigmoidGradient(z2_i)';

  D1_i = D1_i + delta2_i*a1_i;
  D2_i = D2_i + delta3_i*a2_i;

end

lambda_v1 = zeros(size(Theta1));
lambda_v1 = [lambda*eye(size(Theta1, 2))];
lambda_v1(:,1) = 0;

lambda_v2 = zeros(size(Theta2));
lambda_v2 = [lambda*eye(size(Theta2, 2))];
lambda_v2(:,1) = 0;


Theta1_grad = (D1_i + Theta1*lambda_v1)/m;
Theta2_grad = (D2_i + Theta2*lambda_v2)/m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
