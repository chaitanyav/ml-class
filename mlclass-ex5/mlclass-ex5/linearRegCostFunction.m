function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
for i=1:m
    J = J + ((sum(X(i,:)' .* theta) - y(i)) ^ 2);
end
J = (1/(2*m)) * J;

for i=2:size(theta,1)
    J = J + ((lambda/(2*m)) * (theta(i) ^ 2));
end

grad = zeros(size(theta,1), 1);
for i=1:size(theta,1)
    for j=1:m
        grad(i) = grad(i) + ((sum(X(j,:)' .* theta) - y(j)) * X(j,i));
    end
    
    grad(i) = (1/m) * grad(i);
    if(i  > 1)
        grad(i) = grad(i) + ((lambda/m) * theta(i));
    end
end
% =========================================================================
end
