%--------------------------------------------------------------------------
% We would like to show how gradient descent and its variants work.
%
% We assume that the cost function is a quadratic function whose derivative 
% would be easily found. 
%
% Let the cost function be "J(x) = x'*A*x", x' refers to the transpose of x
% the parameter to be optimized is x, which is a vector and whose dimension
% is equal to the dimension of parameter space.
%
% then we use gradient methods to minimize it and show how x moves in the
% parameter space simultaneously.
%
% First we find dJ(x)/dx,
%               dJ(x)/dx = (A + A')x
%
% then update x iteratively until x converge,
%               x := x - ¦Á * dJ(x)/dx
%--------------------------------------------------------------------------
%% plot the contour of J(x) = x'*A*x.
% matrix of quadratic form
A = [0.1, 0;
     0,  5];
% generate a table of function vaule
x_1 = linspace(-2, 2, 50);
x_2 = linspace(-2, 2, 50);
[x1, x2] = meshgrid(x_1, x_2);
J = A(1,1) * x1.^2 + (A(1,2) + A(2,1)) * x1 .* x2 + A(2,2) * x2.^2;
% plot contour of J(x)
figure(1)
contour(x1, x2, J, 20);
axis equal
hold on
grid on
%% parameters
% alpha: learning rate
alpha = 0.195;
% alpha_decay: learning rate decay coefficient (exponentially)
alpha_decay = 1;

% beta: coeff related to 1st-order momentum in GD with Momentum
beta = 0.5;

% beta_RMSprop: coeff related to 2nd-order momentum in RMSprop
beta_RMSprop = 0.99;

% beta_Adam_1: coeff related to 1st-order momentum in Adam
% beta_Adam_2: coeff related to 2nd-order momentum in Adam
beta_Adam_1 = 0.5;
beta_Adam_2 = 0.99;

% initial weight in parameter space
x_init = [1.9; 1.5];

x_GD = [x_init];

x_Momentum = [x_init];
v_Momentum = 0;

x_RMSprop = [x_init];
s_RMSprop = 0;
epsilon = 0;

x_Adam = [x_init];
v_Adam = 0;
s_Adam = 0;

max_iter = 70;
scatter(x_init(1), x_init(2), '.', 'k');
for iter = 1:max_iter
    alpha = alpha_decay * alpha;
    % 1. Standard Gradient Descent
    dx_GD = (A + A') * x_GD(:, end);
    x_GD = [x_GD, x_GD(:, end) - alpha * dx_GD(:, end)];
    
    % 2. Gradient Descent with Momentum
    dx_Momentum = (A + A') * x_Momentum(:, end);
    v_Momentum = beta * v_Momentum + (1-beta) * dx_Momentum;
    x_Momentum = [x_Momentum, x_Momentum(:, end) - alpha * v_Momentum(:, end)];
    
    % 3. RMSprop 
    dx_RMSprop = (A + A') * x_RMSprop(:, end);
    s_RMSprop = beta_RMSprop * s_RMSprop + (1 - beta_RMSprop) * dx_RMSprop.^2;
    x_RMSprop = [x_RMSprop, x_RMSprop(:, end) - alpha * dx_RMSprop(:, end)./sqrt(s_RMSprop + epsilon)];
    
    % 4. Adam
    dx_Adam = (A + A') * x_Adam(:, end);
    v_Adam = beta_Adam_1 * v_Adam + (1 - beta_Adam_1) * dx_Adam;
    s_Adam = beta_Adam_2 * s_Adam + (1 - beta_Adam_2) * dx_Adam.^2;
    x_Adam = [x_Adam, x_Adam(:, end) - alpha * v_Adam./sqrt(s_Adam + epsilon)];
    
    % plot
    figure(1)
    scatter(x_GD(1, end), x_GD(2, end), '.', 'r');
    traj1 = line([x_GD(1, end), x_GD(1, end-1)], [x_GD(2, end), x_GD(2, end-1)], 'color', 'r', 'DisplayName', 'GD');
    
    scatter(x_Momentum(1, end), x_Momentum(2, end), '.', 'b');
    traj2 = line([x_Momentum(1, end), x_Momentum(1, end-1)], [x_Momentum(2, end), x_Momentum(2, end-1)], 'color', 'b', 'DisplayName', 'GD with Momentum');
    
    scatter(x_RMSprop(1, end), x_RMSprop(2, end), '.', 'm');
    traj3 = line([x_RMSprop(1, end), x_RMSprop(1, end-1)], [x_RMSprop(2, end), x_RMSprop(2, end-1)], 'color', 'm', 'DisplayName', 'RMSprop');
    
    scatter(x_Adam(1, end), x_Adam(2, end), '.', 'k');
    traj4 = line([x_Adam(1, end), x_Adam(1, end-1)], [x_Adam(2, end), x_Adam(2, end-1)], 'color', 'k', 'DisplayName', 'Adam');
   
%     legend('show')
    legend([traj1 traj2 traj3 traj4], 'GD', 'GD with Momentum',  'RMSprop', 'Adam', 'Location','northwest')
    xlabel('x1')
    ylabel('x2')
    title(['Iteration: ', num2str(iter)])
    drawnow
    pause(0.05);

end
