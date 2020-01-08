%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m

%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('MVP.txt');
X = data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.

plotData(X, y);


% Put some labels 
hold on;
% Labels and Legend

xlabel('Combined Stats')
ylabel('Win Share')

set(gca,'FontSize',28);

% Specified in plot order
lh= legend('MVP', 'Not MVP');
set(lh, 'FontSize', 20);


hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);


% Plot Boundary
plotDecisionBoundary(theta, X, y);
%

% Put some labels 
hold on;
% Labels and Legend
xlabel('Combined Stats')
ylabel('Win Share')

% Specified in plot order
legend('MVP', 'Not MVP')
hold off;

fprintf('  \n');
fprintf('Whoever has the highest MVP number is the predicted MVP ');
fprintf('  \n');
fprintf('  \n');



% 1 stats WS
fprintf('  \n');
prob = sigmoid([1 61.5 14.4] * theta);
fprintf(['Giannis 2019: %f\n '], prob *100);

fprintf(' ' );
prob = sigmoid([1 53.1328 15.2] * theta);
fprintf(['Harden 2019: %f\n  '], prob *100);



fprintf(' ' );
prob = sigmoid([1 49.9 15.4] * theta);
fprintf(['Harden 2018: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 57.6 14] * theta);
fprintf(['Lebron 2018: %f\n  '], prob *100);


fprintf(' ' );
prob = sigmoid([1 75.35 13.1] * theta);
fprintf(['Westbrook 2017: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 57.19 15] * theta);
fprintf(['Harden 2017: %f\n  '], prob *100);


fprintf(' ' );
prob = sigmoid([1 47.4 17.9] * theta);
fprintf(['Curry 2016: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 38 13.7] * theta);
fprintf(['Kawhi 2016: %f\n  '], prob *100);


fprintf(' ' );
prob = sigmoid([1 42.48 15.7] * theta);
fprintf(['Curry 2015: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 45.4 16.4] * theta);
fprintf(['Harden 2015: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 71 13.1] * theta);
fprintf(['Drose 2014: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 55.53 14.4] * theta);
fprintf(['Bron 2014: %f\n  '], prob *100);

%{
51.3674,6.8,0
61.4135,7,0
86.9723,6.3,0
54.234,8.3,0
%}

fprintf(' ' );
prob = sigmoid([1 51.36 6.8] * theta);
fprintf(['AD 2020: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 62.41 7] * theta);
fprintf(['GF 2020: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 86.9 6.3] * theta);
fprintf(['LUKA 2020: %f\n  '], prob *100);

fprintf(' ' );
prob = sigmoid([1 54.23 8.3] * theta);
fprintf(['HARDEN 2020: %f\n  '], prob *100);




p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);






