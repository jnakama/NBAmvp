clear ; close all; clc

%Load Data

data = load('MVP.txt');
X = data(:, [1, 2]); y = data(:, 3);


%Label and Plot diagram
plotData(X, y);

hold on;
xlabel('Combined Stats')
ylabel('Win Share')

set(gca,'FontSize',28);

lh= legend('MVP', 'Not MVP');
set(lh, 'FontSize', 20);

hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term 
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);


%  options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%this MATLAB function finds optimal thetas
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);


fprintf('  \n');
fprintf('Whoever has the highest MVP number is the predicted MVP ');
fprintf('  \n');
fprintf('  \n');


% printing out examples, and 2020 predictions
fprintf('  \n');
prob = sigmoid([1 61.5 14.4] * theta);
fprintf(['Giannis 2019: %f\n '], prob *100);

fprintf(' ' );
prob = sigmoid([1 53.1328 15.2] * theta);
fprintf(['Harden 2019: %f\n  '], prob *100);



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






