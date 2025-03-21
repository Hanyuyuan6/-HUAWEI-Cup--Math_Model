clc; clear; close all;

%% 加载数据
load('附件2-风电机组采集数据.mat');

%% 加载回归模型和特征均值/标准差
load('mdl_Cp.mat', 'mdl_Cp');         % 加载功率系数 Cp 的回归模型
load('mdl_Ct.mat', 'mdl_Ct');         % 加载推力系数 Ct 的回归模型
load('feature_mean_std.mat', 'feature_mean', 'feature_std');  % 包含 feature_mean 和 feature_std

%% 风机和空气参数定义
rho = 1.225;            % 空气密度 kg/m^3
R = 72.5;               % 风轮半径 m
A = pi * R^2;           % 风轮扫掠面积 m^2

%% 数据准备 - 风电场（WF_1）
wind_farm = data_TS_WF.WF_1.WT;    % 使用 WF_1 数据
num_turbines = 100;     % 风机数量

% 初始化存储计算结果的矩阵
num_time_steps = 100;   % 前100秒
Torque_matrix = NaN(num_time_steps, num_turbines);  % [时间 x 风机]
Thrust_matrix = NaN(num_time_steps, num_turbines);

%% 遍历每台风机，计算扭矩和推力
for turbine = 1:num_turbines
    % 提取当前风机的数据
    wind_speed = wind_farm{turbine}.inputs(:,2);      % 风速 V
    power_pref = wind_farm{turbine}.inputs(:,1);      % 功率参考值 Pref
    power_out = wind_farm{turbine}.outputs(:,3);      % 实际输出功率 P_out
    pitch_angle = wind_farm{turbine}.states(:,1);     % 桨距角 β
    omega_r = wind_farm{turbine}.states(:,2);         % 转速 ω_r

    % 取前100个数据点
    wind_speed = wind_speed(1:num_time_steps);
    power_pref = power_pref(1:num_time_steps);
    power_out = power_out(1:num_time_steps);
    pitch_angle = pitch_angle(1:num_time_steps);
    omega_r = omega_r(1:num_time_steps);

    % 数据预处理 - 去除异常值和缺失值
    valid_rows = ~any(isnan([wind_speed, power_pref, power_out, pitch_angle, omega_r]), 2) & ...
                 ~any(isinf([wind_speed, power_pref, power_out, pitch_angle, omega_r]), 2) & ...
                 (wind_speed > 0);

    % 如果所有数据都是无效的，跳过此风机
    if all(~valid_rows)
        continue;
    end

    % 仅保留有效的数据
    wind_speed = wind_speed(valid_rows);
    power_pref = power_pref(valid_rows);
    power_out = power_out(valid_rows);
    pitch_angle = pitch_angle(valid_rows);
    omega_r = omega_r(valid_rows);

    % 计算叶尖速比 λ
    lambda = omega_r .* R ./ wind_speed;  % λ = ω * R / V

    % 计算功率差值
    power_diff = power_pref - power_out;

    % 构建特征矩阵
    data_turbine = [lambda, pitch_angle, power_pref, power_diff];

    % 对特征进行标准化
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
    T_estimated_turbine(omega_m == 0) = NaN;  % 避免除以零
    T_estimated_turbine(isnan(T_estimated_turbine) | isinf(T_estimated_turbine)) = NaN;

    % 计算估算的塔架推力 F_estimated_turbine
    F_estimated_turbine = 0.5 * rho * A .* Ct_predicted_turbine .* wind_speed.^2;

    % 将计算结果存入矩阵
    Torque_matrix(valid_rows, turbine) = T_estimated_turbine;
    Thrust_matrix(valid_rows, turbine) = F_estimated_turbine;
end

% 保存扭矩和推力矩阵
save('Torque_Thrust_Matrices.mat', 'Torque_matrix', 'Thrust_matrix');

% 显示完成信息
disp('扭矩和推力计算完成，结果已保存到 Torque_Thrust_Matrices.mat');