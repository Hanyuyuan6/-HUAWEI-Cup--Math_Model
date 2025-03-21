clc; clear; close all;

%% 加载数据
load('附件2-风电机组采集数据.mat');

%% 风机和空气参数定义
rho = 1.225;            % 空气密度 kg/m^3
R = 72.5;                 % 风轮半径 m
A = pi * R^2;           % 风轮扫掠面积 m^2

%% 数据准备 - 训练集（WF_2）
% 从 WF_2 中提取数据，用于训练模型
wind_farm_train = data_TS_WF.WF_2.WT;    % 训练集
num_turbines_train = 100;     % 训练集风机数量

% 初始化训练集的数据矩阵和目标变量
data_matrix_train = [];      % 训练集特征变量
Cp_targets_train = [];       % 训练集功率系数 Cp 目标值
Ct_targets_train = [];       % 训练集推力系数 Ct 目标值
torque_targets_train = [];   % 训练集主轴扭矩目标值
thrust_targets_train = [];   % 训练集塔架推力目标值

%% 遍历每台风机，收集训练集数据
for turbine = 1:num_turbines_train
    % 提取当前风机的数据
    wind_speed = wind_farm_train{turbine}.inputs(:,2);      % 风速 V (2000×1)
    power_pref = wind_farm_train{turbine}.inputs(:,1);      % 功率参考值 Pref (2000×1)
    power_out = wind_farm_train{turbine}.outputs(:,3);      % 实际输出功率 P_out (2000×1)
    pitch_angle = wind_farm_train{turbine}.states(:,1);     % 桨距角 β (2000×1)
    omega_r = wind_farm_train{turbine}.states(:,2);         % 转速 ω_r (2000×1)
    torque_ref = wind_farm_train{turbine}.outputs(:,1);     % 主轴扭矩参考值 T_shaft (2000×1)
    thrust_ref = wind_farm_train{turbine}.outputs(:,2);     % 塔架推力参考值 F_thrust (2000×1)

    % 数据预处理 - 去除异常值和缺失值
    turbine_data = [wind_speed, power_pref, power_out, pitch_angle, omega_r, torque_ref, thrust_ref];
    valid_rows = ~any(isnan(turbine_data), 2) & ~any(isinf(turbine_data), 2);
    turbine_data = turbine_data(valid_rows, :);

    % 提取变量
    wind_speed = turbine_data(:,1);
    power_pref = turbine_data(:,2);
    power_out = turbine_data(:,3);
    pitch_angle = turbine_data(:,4);
    omega_r = turbine_data(:,5);
    torque_ref = turbine_data(:,6);
    thrust_ref = turbine_data(:,7);

    % 计算叶尖速比 λ
    lambda = omega_r .* R ./ wind_speed;  % λ = ω * R / V

    % 计算功率差值
    power_diff = power_pref - power_out;

    % 构建特征矩阵
    data_train = [lambda, pitch_angle, power_pref, power_diff];

    % 计算功率系数 Cp_actual
    Cp_actual = power_out ./ (0.5 * rho * A .* wind_speed.^3);
    Cp_actual(Cp_actual < 0 | Cp_actual > 1) = NaN;

    % 计算推力系数 Ct_actual
    Ct_actual = thrust_ref ./ (0.5 * rho * A .* wind_speed.^2);
    Ct_actual(Ct_actual < 0 | Ct_actual > 1) = NaN;

    % 去除异常值
    valid_indices = all(~isnan([Cp_actual, Ct_actual, data_train]), 2);
    data_train = data_train(valid_indices, :);
    Cp_actual = Cp_actual(valid_indices);
    Ct_actual = Ct_actual(valid_indices);
    torque_ref = torque_ref(valid_indices);
    thrust_ref = thrust_ref(valid_indices);

    % 累积数据
    data_matrix_train = [data_matrix_train; data_train];
    Cp_targets_train = [Cp_targets_train; Cp_actual];
    Ct_targets_train = [Ct_targets_train; Ct_actual];
    torque_targets_train = [torque_targets_train; torque_ref];
    thrust_targets_train = [thrust_targets_train; thrust_ref];
end

%% 数据准备 - 测试集（WF_1）
% 从 WF_1 中提取数据，用于测试模型
wind_farm_test = data_TS_WF.WF_1.WT;    % 测试集
num_turbines_test = 100;     % 测试集风机数量

% 初始化测试集的数据矩阵和目标变量
data_matrix_test = [];       % 测试集特征变量
torque_targets_test = [];    % 测试集主轴扭矩目标值
thrust_targets_test = [];    % 测试集塔架推力目标值
omega_r_test_all = [];       % 测试集转速
power_out_test_all = [];     % 测试集实际输出功率
power_pref_test_all = [];    % 测试集功率参考值
wind_speed_test_all = [];    % 测试集风速
Cp_targets_test = [];        % 测试集功率系数 Cp 目标值
Ct_targets_test = [];        % 测试集推力系数 Ct 目标值

%% 遍历每台风机，收集测试集数据
for turbine = 1:num_turbines_test
    % 提取当前风机的数据
    wind_speed = wind_farm_test{turbine}.inputs(:,2);      % 风速 V (2000×1)
    power_pref = wind_farm_test{turbine}.inputs(:,1);      % 功率参考值 Pref (2000×1)
    power_out = wind_farm_test{turbine}.outputs(:,3);      % 实际输出功率 P_out (2000×1)
    pitch_angle = wind_farm_test{turbine}.states(:,1);     % 桨距角 β (2000×1)
    omega_r = wind_farm_test{turbine}.states(:,2);         % 转速 ω_r (2000×1)
    torque_ref = wind_farm_test{turbine}.outputs(:,1);     % 主轴扭矩参考值 T_shaft (2000×1)
    thrust_ref = wind_farm_test{turbine}.outputs(:,2);     % 塔架推力参考值 F_thrust (2000×1)

    % 数据预处理 - 去除异常值和缺失值
    turbine_data = [wind_speed, power_pref, power_out, pitch_angle, omega_r, torque_ref, thrust_ref];
    valid_rows = ~any(isnan(turbine_data), 2) & ~any(isinf(turbine_data), 2);
    turbine_data = turbine_data(valid_rows, :);

    % 提取变量
    wind_speed = turbine_data(:,1);
    power_pref = turbine_data(:,2);
    power_out = turbine_data(:,3);
    pitch_angle = turbine_data(:,4);
    omega_r = turbine_data(:,5);
    torque_ref = turbine_data(:,6);
    thrust_ref = turbine_data(:,7);

    % 计算叶尖速比 λ
    lambda = omega_r .* R ./ wind_speed;  % λ = ω * R / V

    % 计算功率差值
    power_diff = power_pref - power_out;

    % 构建特征矩阵
    data_test = [lambda, pitch_angle, power_pref, power_diff];

    % 计算功率系数 Cp_actual
    Cp_actual = power_out ./ (0.5 * rho * A .* wind_speed.^3);
    Cp_actual(Cp_actual < 0 | Cp_actual > 1) = NaN;

    % 计算推力系数 Ct_actual
    Ct_actual = thrust_ref ./ (0.5 * rho * A .* wind_speed.^2);
    Ct_actual(Ct_actual < 0 | Ct_actual > 1) = NaN;

    % 去除异常值
    valid_indices = all(~isnan([Cp_actual, Ct_actual, data_test]), 2);
    data_test = data_test(valid_indices, :);
    torque_ref = torque_ref(valid_indices);
    thrust_ref = thrust_ref(valid_indices);
    omega_r = omega_r(valid_indices);
    power_out = power_out(valid_indices);
    power_pref = power_pref(valid_indices);
    wind_speed = wind_speed(valid_indices);
    Cp_actual = Cp_actual(valid_indices);
    Ct_actual = Ct_actual(valid_indices);

    % 累积数据
    data_matrix_test = [data_matrix_test; data_test];
    torque_targets_test = [torque_targets_test; torque_ref];
    thrust_targets_test = [thrust_targets_test; thrust_ref];
    omega_r_test_all = [omega_r_test_all; omega_r];
    power_out_test_all = [power_out_test_all; power_out];
    power_pref_test_all = [power_pref_test_all; power_pref];
    wind_speed_test_all = [wind_speed_test_all; wind_speed];
    Cp_targets_test = [Cp_targets_test; Cp_actual];
    Ct_targets_test = [Ct_targets_test; Ct_actual];
end

%% 特征工程
% 构建训练集特征矩阵
features_train = data_matrix_train;

% 对特征进行标准化
feature_mean = mean(features_train);
feature_std = std(features_train);
features_train_standardized = (features_train - feature_mean) ./ feature_std;

% 使用多元线性回归拟合 Cp 和 Ct
mdl_Cp = fitlm(features_train_standardized, Cp_targets_train);
mdl_Ct = fitlm(features_train_standardized, Ct_targets_train);

% 显示模型摘要
disp('功率系数 Cp 回归模型摘要：');
disp(mdl_Cp);

disp('推力系数 Ct 回归模型摘要：');
disp(mdl_Ct);

% 计算训练集上的 R² 值
R2_Cp_train = mdl_Cp.Rsquared.Adjusted;
disp(['训练集功率系数 Cp 模型的调整后 R² 值：', num2str(R2_Cp_train)]);

R2_Ct_train = mdl_Ct.Rsquared.Adjusted;
disp(['训练集推力系数 Ct 模型的调整后 R² 值：', num2str(R2_Ct_train)]);

%% 在测试集上进行预测
% 标准化测试集特征（使用训练集的均值和标准差）
features_test_standardized = (data_matrix_test - feature_mean) ./ feature_std;

% 预测测试集的 Cp 和 Ct
Cp_predicted_test = predict(mdl_Cp, features_test_standardized);
Ct_predicted_test = predict(mdl_Ct, features_test_standardized);

% 限制 Cp 和 Ct 的值在合理范围内
Cp_predicted_test(Cp_predicted_test < 0) = 0;
Cp_predicted_test(Cp_predicted_test > 1) = 1;
Ct_predicted_test(Ct_predicted_test < 0) = 0;
Ct_predicted_test(Ct_predicted_test > 1) = 1;

% 计算估算的输出功率 P_estimated_test
wind_speed_test = wind_speed_test_all;
P_estimated_test = 0.5 * rho * A .* Cp_predicted_test .* wind_speed_test.^3;

% 计算估算的主轴扭矩 T_estimated_test
omega_m_test = omega_r_test_all;  % 假设无齿轮箱
T_estimated_test = P_estimated_test ./ omega_m_test;
T_estimated_test(isnan(T_estimated_test) | isinf(T_estimated_test)) = 0;

% 计算估算的塔架推力 F_estimated_test
F_estimated_test = 0.5 * rho * A .* Ct_predicted_test .* wind_speed_test.^2;

%% 评估模型性能

% 计算测试集上的误差平方和
error_thrust_test = sum((F_estimated_test - thrust_targets_test).^2);
error_torque_test = sum((T_estimated_test - torque_targets_test).^2);

disp(['测试集塔架推力总误差平方和：', num2str(error_thrust_test)]);
disp(['测试集主轴扭矩总误差平方和：', num2str(error_torque_test)]);

% 计算测试集上的 R² 值
SS_res_torque_test = sum((torque_targets_test - T_estimated_test).^2);
SS_tot_torque_test = sum((torque_targets_test - mean(torque_targets_test)).^2);
R2_torque_test = 1 - (SS_res_torque_test / SS_tot_torque_test);
disp(['测试集主轴扭矩模型的 R² 值：', num2str(R2_torque_test)]);

SS_res_thrust_test = sum((thrust_targets_test - F_estimated_test).^2);
SS_tot_thrust_test = sum((thrust_targets_test - mean(thrust_targets_test)).^2);
R2_thrust_test = 1 - (SS_res_thrust_test / SS_tot_thrust_test);
disp(['测试集塔架推力模型的 R² 值：', num2str(R2_thrust_test)]);

%% 对单台风机的数据进行验证
% 以测试集中的第一台风机为例
turbine = 1;  % 测试集风机编号

% 提取该风机的数据
wind_speed = wind_farm_test{turbine}.inputs(:,2);      % 风速 V
power_pref = wind_farm_test{turbine}.inputs(:,1);      % 功率参考值 Pref
power_out = wind_farm_test{turbine}.outputs(:,3);      % 实际输出功率 P_out
pitch_angle = wind_farm_test{turbine}.states(:,1);     % 桨距角 β
omega_r = wind_farm_test{turbine}.states(:,2);         % 转速 ω_r
torque_ref = wind_farm_test{turbine}.outputs(:,1);     % 主轴扭矩参考值
thrust_ref = wind_farm_test{turbine}.outputs(:,2);     % 塔架推力参考值

% 数据预处理
turbine_data = [wind_speed, power_pref, power_out, pitch_angle, omega_r, torque_ref, thrust_ref];
valid_rows = ~any(isnan(turbine_data), 2) & ~any(isinf(turbine_data), 2);
turbine_data = turbine_data(valid_rows, :);

% 提取变量
wind_speed = turbine_data(:,1);
power_pref = turbine_data(:,2);
power_out = turbine_data(:,3);
pitch_angle = turbine_data(:,4);
omega_r = turbine_data(:,5);
torque_ref = turbine_data(:,6);
thrust_ref = turbine_data(:,7);

% 计算叶尖速比 λ
lambda = omega_r .* R ./ wind_speed;  % λ = ω * R / V

% 计算功率差值
power_diff = power_pref - power_out;

% 构建特征矩阵
data_turbine = [lambda, pitch_angle, power_pref, power_diff];

% 标准化特征
features_turbine_standardized = (data_turbine - feature_mean) ./ feature_std;

% 预测 Cp 和 Ct
Cp_predicted_turbine = predict(mdl_Cp, features_turbine_standardized);
Ct_predicted_turbine = predict(mdl_Ct, features_turbine_standardized);

% 限制 Cp 和 Ct 的值在合理范围内
Cp_predicted_turbine(Cp_predicted_turbine < 0) = 0;
Cp_predicted_turbine(Cp_predicted_turbine > 1) = 1;
Ct_predicted_turbine(Ct_predicted_turbine < 0) = 0;
Ct_predicted_turbine(Ct_predicted_turbine > 1) = 1;

% 计算估算的输出功率 P_estimated_turbine
P_estimated_turbine = 0.5 * rho * A .* Cp_predicted_turbine .* wind_speed.^3;

% 计算估算的主轴扭矩 T_estimated_turbine
omega_m = omega_r;  % 假设无齿轮箱
T_estimated_turbine = P_estimated_turbine ./ omega_m;
T_estimated_turbine(isnan(T_estimated_turbine) | isinf(T_estimated_turbine)) = 0;

% 计算估算的塔架推力 F_estimated_turbine
F_estimated_turbine = 0.5 * rho * A .* Ct_predicted_turbine .* wind_speed.^2;

% 计算 R² 值
SS_res_torque_turbine = sum((torque_ref - T_estimated_turbine).^2);
SS_tot_torque_turbine = sum((torque_ref - mean(torque_ref)).^2);
R2_torque_turbine = 1 - (SS_res_torque_turbine / SS_tot_torque_turbine);
disp(['风机 ', num2str(turbine), ' 的主轴扭矩模型 R² 值：', num2str(R2_torque_turbine)]);

SS_res_thrust_turbine = sum((thrust_ref - F_estimated_turbine).^2);
SS_tot_thrust_turbine = sum((thrust_ref - mean(thrust_ref)).^2);
R2_thrust_turbine = 1 - (SS_res_thrust_turbine / SS_tot_thrust_turbine);
disp(['风机 ', num2str(turbine), ' 的塔架推力模型 R² 值：', num2str(R2_thrust_turbine)]);

% 绘制结果
time = (1:length(torque_ref))';

figure;
subplot(2,1,1);
plot(time, torque_ref, 'b', 'LineWidth', 1.5);
hold on;
plot(time, T_estimated_turbine, 'r--', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('主轴扭矩 (Nm)');
title(['测试集风机 ', num2str(turbine), ' 的主轴扭矩估算值与实际值']);
legend('实际值', '估算值');
grid on;

subplot(2,1,2);
plot(time, thrust_ref, 'b', 'LineWidth', 1.5);
hold on;
plot(time, F_estimated_turbine, 'r--', 'LineWidth', 1.5);
xlabel('时间 (s)');
ylabel('塔架推力 (N)');
title(['测试集风机 ', num2str(turbine), ' 的塔架推力估算值与实际值']);
legend('实际值', '估算值');
grid on;