clc; clear; close all;

%% 加载数据
load('附件2-风电机组采集数据.mat');

%% 风机参数定义
rho = 1.225;            % 空气密度 kg/m^3
R = 72.5;                % 风轮半径 m
A = pi * R^2;           % 风轮扫掠面积 m^2
V_rated = 11.2;         % 额定风速 m/s

%% 设置风电场
% 选择要分析的风电场数据，包括 WF_1 或 WF_2
% wind_farm = data_TS_WF.WF_1.WT;  % 选择 WF_1 风电场数据
wind_farm = data_TS_WF.WF_2.WT;    % 如果需要选择 WF_2，取消注释此行

num_turbines = 100;     % 风机数量
total_time = 2000;      % 总时间（秒）

%% 初始化结果矩阵
thrust_estimated = zeros(total_time, num_turbines);
torque_estimated = zeros(total_time, num_turbines);

%% 辅助函数：计算功率系数Cp和推力系数Ct
% 计算Cp与Ct
Cp_max = 0.593;  % 贝茨极限，Cp的理论最大值

% 定义用于计算Cp的函数
Cp_function = @(lambda, beta) 0.5176 * ((116 ./ (1 ./ (lambda + 0.08 .* beta) - 0.035 ./ (beta.^3 + 1))) - 0.45 .* beta - 5) ...
                             .* exp(-21 ./ (1 ./ (lambda + 0.08 .* beta) - 0.035 ./ (beta.^3 + 1))) ...
                             + 0.0068 .* lambda;

% 叶尖速比函数
lambda_function = @(omega_m, R, v) omega_m * R ./ v;

% 推力系数与功率系数的关系
Ct_function = @(Cp) 0.5 * Cp / Cp_max;

%% 处理每台风机的数据
for turbine = 1:num_turbines
    % 提取当前风机的数据
    wind_speed = wind_farm{turbine}.inputs(:,2);      % 提取风速 V (2000×1)
    power_pref = wind_farm{turbine}.inputs(:,1);      % 提取功率调度指令（Pref） (2000×1)
    pitch_angle = wind_farm{turbine}.states(:,1);     % 提取桨距角 β (2000×1)
    omega_r = wind_farm{turbine}.states(:,2);         % 低速轴转速 ω (2000×1)
    power_out = wind_farm{turbine}.outputs(:,3);      % 实际输出功率 (2000×1)
    
    % 确保数据长度一致
    n = length(wind_speed);
    
    % 计算叶尖速比 λ
    lambda = lambda_function(omega_r, R, wind_speed);
    
    % 估算功率系数 Cp
    Cp_estimated = Cp_function(lambda, pitch_angle);
    Cp_estimated(isnan(Cp_estimated) | Cp_estimated < 0) = 0;
    Cp_estimated(Cp_estimated > Cp_max) = Cp_max;
    
    % 估算推力系数 Ct
    Ct_estimated = Ct_function(Cp_estimated);
    
    % 计算推力 T
    thrust = 0.5 * rho * A .* Ct_estimated .* wind_speed.^2;  % (n×1)
    thrust_estimated(1:n, turbine) = thrust;
    
    % 计算主轴扭矩 M
    torque = power_pref ./ (omega_r + 1e-6);  % 防止除以零
    torque_estimated(1:n, turbine) = torque;
end

%% 计算估算值与参考值之差的平方和以及 R²
error_thrust = zeros(num_turbines, 1);
error_torque = zeros(num_turbines, 1);
R2_thrust = zeros(num_turbines, 1);
R2_torque = zeros(num_turbines, 1);

for turbine = 1:num_turbines
    thrust_ref = wind_farm{turbine}.outputs(:,2);  % 实际塔架推力参考值 (2000×1)
    torque_ref = wind_farm{turbine}.outputs(:,1);  % 实际主轴扭矩参考值 (2000×1)
    
    % 计算误差平方和
    error_thrust(turbine) = sum((thrust_estimated(:, turbine) - thrust_ref).^2);
    error_torque(turbine) = sum((torque_estimated(:, turbine) - torque_ref).^2);
    
    % 计算 R²
    SS_res_thrust = sum((thrust_ref - thrust_estimated(:, turbine)).^2);   % 残差平方和
    SS_tot_thrust = sum((thrust_ref - mean(thrust_ref)).^2);  % 总平方和
    R2_thrust(turbine) = 1 - (SS_res_thrust / SS_tot_thrust); % R² 公式
    
    SS_res_torque = sum((torque_ref - torque_estimated(:, turbine)).^2);   % 残差平方和
    SS_tot_torque = sum((torque_ref - mean(torque_ref)).^2);  % 总平方和
    R2_torque(turbine) = 1 - (SS_res_torque / SS_tot_torque); % R² 公式
end

% 输出误差和 R² 结果
disp('塔架推力估算误差平方和（每台风机）：');
disp(error_thrust);

disp('主轴扭矩估算误差平方和（每台风机）：');
disp(error_torque);

disp('塔架推力 R²（每台风机）：');
disp(R2_thrust);

disp('主轴扭矩 R²（每台风机）：');
disp(R2_torque);

%% 结果可视化
turbines_to_plot = [1, 5, 10];  % 可以根据需要选择不同的风机编号
time = (1:total_time)';  % 时间序列

for i = 1:length(turbines_to_plot)
    turbine = turbines_to_plot(i);
    
    thrust_ref = wind_farm{turbine}.outputs(:,2);  % 实际塔架推力参考值
    torque_ref = wind_farm{turbine}.outputs(:,1);  % 实际主轴扭矩参考值
    
    figure;
    subplot(2, 1, 1);  % 创建子图1
    plot(time, thrust_estimated(:, turbine), 'b', 'LineWidth', 1.5);
    hold on;
    plot(time, thrust_ref, 'r--', 'LineWidth', 1.5);
    xlabel('时间 (s)');
    ylabel('塔架推力 (N)');
    title(['风机 ', num2str(turbine), ' 的塔架推力估算值与参考值']);
    legend('估算值', '参考值');
    grid on;
    
    subplot(2, 1, 2);  % 创建子图2
    plot(time, torque_estimated(:, turbine), 'b', 'LineWidth', 1.5);
    hold on;
    plot(time, torque_ref, 'r--', 'LineWidth', 1.5);
    xlabel('时间 (s)');
    ylabel('主轴扭矩 (Nm)');
    title(['风机 ', num2str(turbine), ' 的主轴扭矩估算值与参考值']);
    legend('估算值', '参考值');
    grid on;
end