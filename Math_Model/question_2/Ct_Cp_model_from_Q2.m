% 问题二：训练并保存模型

clc; clear; close all;

%% 加载数据
% 请确保附件2的数据文件 '附件2-风电机组采集数据.mat' 位于当前工作目录
load('附件2-风电机组采集数据.mat');

%% 风机和空气参数定义
rho = 1.225;            % 空气密度 kg/m^3
R = 65;                 % 风轮半径 m
A = pi * R^2;           % 风轮扫掠面积 m^2

%% 数据准备 - 训练集（WF_2）
% 从 WF_2 中提取数据，用于训练模型
wind_farm_train = data_TS_WF.WF_2.WT;    % 训练集
num_turbines_train = 100;     % 训练集风机数量

% 初始化训练集的数据矩阵和目标变量
data_matrix_train = [];      % 训练集特征变量
Cp_targets_train = [];       % 训练集功率系数 Cp 目标值
Ct_targets_train = [];       % 训练集推力系数 Ct 目标值

%% 遍历每台风机，收集训练集数据
for turbine = 1:num_turbines_train
    % 提取当前风机的数据
    wind_speed = wind_farm_train{turbine}.inputs(:,2);      % 风速 V (2000×1)
    power_pref = wind_farm_train{turbine}.inputs(:,1);      % 功率参考值 Pref (2000×1)
    power_out = wind_farm_train{turbine}.outputs(:,3);      % 实际输出功率 P_out (2000×1)
    pitch_angle = wind_farm_train{turbine}.states(:,1);     % 桨距角 β (2000×1)
    omega_r = wind_farm_train{turbine}.states(:,2);         % 转速 ω_r (2000×1)
    thrust_ref = wind_farm_train{turbine}.outputs(:,2);     % 塔架推力参考值 F_thrust (2000×1)

    % 数据预处理 - 去除异常值和缺失值
    turbine_data = [wind_speed, power_pref, power_out, pitch_angle, omega_r, thrust_ref];
    valid_rows = ~any(isnan(turbine_data), 2) & ~any(isinf(turbine_data), 2);
    turbine_data = turbine_data(valid_rows, :);

    % 提取变量
    wind_speed = turbine_data(:,1);
    power_pref = turbine_data(:,2);
    power_out = turbine_data(:,3);
    pitch_angle = turbine_data(:,4);
    omega_r = turbine_data(:,5);
    thrust_ref = turbine_data(:,6);

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

    % 累积数据
    data_matrix_train = [data_matrix_train; data_train];
    Cp_targets_train = [Cp_targets_train; Cp_actual];
    Ct_targets_train = [Ct_targets_train; Ct_actual];
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

% 保存模型和特征均值、标准差
save('mdl_Cp.mat', 'mdl_Cp');
save('mdl_Ct.mat', 'mdl_Ct');
save('feature_mean_std.mat', 'feature_mean', 'feature_std');

% 提示模型已保存
disp('模型mdl_Cp和mdl_Ct已保存，特征的均值和标准差已保存。');