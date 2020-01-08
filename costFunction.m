function [J, grad] = costFunction(theta, X, y)


m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));

%compute cost and gradient descent

Hx = sigmoid(X * theta);

J =  (1/m) * (sum (-y .* log (Hx) - (1 - y) .* log(1 - Hx) ));

grad =  (1/m) * ((X' * (Hx - y)));




end
