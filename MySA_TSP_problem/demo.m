%% demo
% 模拟退火解决TSP问题
%

%% 随机产生城市坐标
rand('seed', 0);
N_cities = 30;
city_coord = rand(N_cities, 2);
% city_coord = [city_coord; city_coord(1, :)];
idx_init = randperm(N_cities);
idx_init = [idx_init, idx_init(1)];
fig = figure();
set(fig,'units','normalized','position',[0.1 0.1 0.5 0.3]);
plot_cities(city_coord, idx_init, fig);

%% 模拟退火优化
T_start = 1e2;
T = T_start;
T_end = 1;
decay = 0.999;
k = 500;

sol_now = idx_init;
dist_now = calc_dist(city_coord, sol_now);

sol_new = idx_init;
dist_new = calc_dist(city_coord, sol_new);

sol_best = idx_init;
dist_best = calc_dist(city_coord, sol_best);

dist_hist = dist_now;
idx_template = 2:N_cities;
iter = 0;
while(T >= T_end)
    iter = iter + 1;
    swap_idx = idx_template(randperm(N_cities-1, 2));
    
    sol_new = swap_cities(sol_now, swap_idx);
    dist_new = calc_dist(city_coord, sol_new);
    
    if dist_new < dist_now
        sol_now = sol_new;
        dist_now = dist_new;
        sol_best = sol_new;
        dist_best = dist_new;
    elseif rand < exp(k*(dist_now - dist_new)/T)
        fprintf("th = %.3f\n", exp(k*(dist_now - dist_new)/T));
        sol_now = sol_new;
        dist_now = dist_new;
    end
    
    plot_cities(city_coord, sol_now, fig);
    
    % 1. alter 1 exponentially decay
    T = T * decay;
    
    % 2. linearly decay
    % T = T_start * (1 - iter * (1 - decay)) ;
    
    dist_hist = [dist_hist, dist_now];   
    figure(fig);
    subplot(122);
    plot(dist_hist);
 
end






%% 交换城市顺序
function idx = swap_cities(idx, swap_idx)
temp = idx(swap_idx(1));
idx(swap_idx(1)) = idx(swap_idx(2));
idx(swap_idx(2)) = temp;
end

%% 绘图
function plot_cities(city_coord, idx, fig)

figure(fig);
subplot(121);
city_coord = city_coord(idx, :);
plot(city_coord(:, 1), city_coord(:, 2));

end

%% 计算距离
function dist = calc_dist(city_coord, idx)
city_coord = city_coord(idx, :);
delta = city_coord(2:end, :) - city_coord(1:end-1, :);
dist = sum(sqrt(delta(:, 1).^2 + delta(:, 2).^2));
end

