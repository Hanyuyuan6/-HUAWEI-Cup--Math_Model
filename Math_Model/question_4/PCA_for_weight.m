clc; clear; close all;

%% 加载数据
load('附件2-风电机组采集数据.mat');

%% 风机和空气参数定义
rho = 1.225;         % 空气密度 kg/m^3
R = 72.5;            % 风轮半径 m
A = pi * R^2;        % 风轮扫掠面积 m^2

%% 数据准备
% 从 WF_2 中提取数据
wind_farm_train = data_TS_WF.WF_2.WT; 
num_turbines_train = 100;     % 风机数量

% 初始化数据矩阵和目标变量
data_matrix_train = [];       % 特征变量，用于回归模型训练
Cp_targets_train = [];        % 功率系数 Cp 目标值
Ct_targets_train = [];        % 推力系数 Ct 目标值
torque_targets_train = [];    % 主轴扭矩目标值
thrust_targets_train = [];    % 塔架推力目标值

% 初始化变量，用于 PCA
wind_speed_all = [];
pitch_angle_all = [];
omega_r_all = [];
power_out_all = [];
torque_ref_all = [];
thrust_ref_all = [];

%% 遍历每台风机
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
    wind_speed = wind_speed(valid_indices);
    pitch_angle = pitch_angle(valid_indices);
    omega_r = omega_r(valid_indices);
    power_out = power_out(valid_indices);

    % 累积数据
    data_matrix_train = [data_matrix_train; data_train];
    Cp_targets_train = [Cp_targets_train; Cp_actual];
    Ct_targets_train = [Ct_targets_train; Ct_actual];
    torque_targets_train = [torque_targets_train; torque_ref];
    thrust_targets_train = [thrust_targets_train; thrust_ref];

    % 累积数据用于 PCA
    wind_speed_all = [wind_speed_all; wind_speed];
    pitch_angle_all = [pitch_angle_all; pitch_angle];
    omega_r_all = [omega_r_all; omega_r];
    power_out_all = [power_out_all; power_out];
    torque_ref_all = [torque_ref_all; torque_ref];
    thrust_ref_all = [thrust_ref_all; thrust_ref];
end

%% PCA 分析，得到变量权重
% 1. 组织数据矩阵
data_matrix = [wind_speed_all, pitch_angle_all, omega_r_all, power_out_all, torque_ref_all, thrust_ref_all];

% 2. 数据标准化
data_standardized = (data_matrix - mean(data_matrix)) ./ std(data_matrix);

% 3. 执行 PCA
[coeff, score, latent, tsquared, explained] = pca(data_standardized);

% 4. 查看主成分和解释的方差比例
disp('各主成分的方差解释比例 (%):');
disp(explained);

% 5. 查看第一主成分的载荷 (loadings)
variable_names = {'Wind Speed', 'Pitch Angle', 'Rotor Speed', 'Power Output', 'Torque', 'Thrust'};
loadings_PC1 = coeff(:,1);

% 显示变量在第一主成分上的载荷
fprintf('第一主成分的载荷:\n');
for i = 1:length(variable_names)
    fprintf('%s: %f\n', variable_names{i}, loadings_PC1(i));
end

% 6. 计算权重
% 我们采用第一主成分的载荷的绝对值作为变量的重要性指标
abs_loadings_PC1 = abs(loadings_PC1);
weights = abs_loadings_PC1 / sum(abs_loadings_PC1);

% 显示变量权重
fprintf('基于第一主成分的变量权重:\n');
for i = 1:length(variable_names)
    fprintf('%s: %f\n', variable_names{i}, weights(i));
end

