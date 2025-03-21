%% 清除环境
clc; clear; close all;

%% 加载数据
% 从数据文件中加载噪声和延迟作用下风电机组数据 (附件3: 训练数据)
load('附件3-噪声和延迟作用下的采集数据.mat'); % 附件3-噪声和延迟作用下的采集数据为 data_TS_WF_noise

% 加载附件4的数据：额外的十台风机的 300s 测量数据, 作为测试集检验鲁棒模型
filename_test = '附件4-噪声和延迟作用下的采集数据.xlsx';  % Excel 文件名
startRow = 10;     % 开始读取的行
numRows = 100;     % 读取的数据行数
num_turbines_test_used = 10; % 测试数据中的风机数量

% 使用 readmatrix 从 Excel 文件中读取数据（读取全部10列对应10台风机）
range_test = sprintf('A%d:J%d', startRow, startRow + numRows - 1); % 假设10台风机对应A到J列

Ft_test = readmatrix(filename_test, 'Sheet', 'Ft', 'Range', range_test); % 桨距角
Pout_test = readmatrix(filename_test, 'Sheet', 'Pout', 'Range', range_test); % 输出功率
Pref_test = readmatrix(filename_test, 'Sheet', 'Pref', 'Range', range_test); % 调度指令
Vwin_test = readmatrix(filename_test, 'Sheet', 'Vwin', 'Range', range_test); % 风速
wgenM_test = readmatrix(filename_test, 'Sheet', 'wgenM', 'Range', range_test); % 发电机转速

%% 设置参数
total_time = numRows; % 总时间（秒）

% 设置风电场参数 (训练数据中的100台风机)
wind_farm_train = data_TS_WF_noise.WF_1.WT; % 选择 WF_1 风电场的数据
num_turbines_train = length(wind_farm_train); % 风机数量，假设为100

% 初始化训练数据矩阵（用于训练ARIMA模型）
power_schedule_train = zeros(total_time, num_turbines_train); % 调度指令数据 
wind_speed_train = zeros(total_time, num_turbines_train); % 风速矩阵 
pitch_angle_train = zeros(total_time, num_turbines_train); % 桨距角矩阵 
omega_r_train = zeros(total_time, num_turbines_train); % 转速矩阵 
power_out_train = zeros(total_time, num_turbines_train); % 实际输出功率矩阵

% 遍历每台风机，提取相应的数据（取前 numRows 行数据）
for turbine = 1:num_turbines_train
    power_schedule_train(:, turbine) = wind_farm_train{turbine}.inputs(1:total_time,1); % 提取调度指令数据 Pref
    wind_speed_train(:, turbine) = wind_farm_train{turbine}.inputs(1:total_time,2); % 提取风速数据
    pitch_angle_train(:, turbine) = wind_farm_train{turbine}.states(1:total_time,1); % 提取桨距角数据
    omega_r_train(:, turbine) = wind_farm_train{turbine}.states(1:total_time,2); % 提取转速数据
    power_out_train(:, turbine) = wind_farm_train{turbine}.outputs(1:total_time,3); % 提取实际输出功率数据
end

% 将测试集中10台风机的数据直接使用
power_schedule_test = Pref_test; % 调度指令 Pref
wind_speed_test = Vwin_test; % 风速
pitch_angle_test = Ft_test; % 桨距角
omega_r_test = wgenM_test; % 转速
power_out_test = Pout_test; % 输出功率

%% 初始化优化相关参数
P_max = 5e6; % 风机额定功率为 5MW (单位：瓦特)
Delta_P_max = 1e6; % 功率波动约束为 1MW (单位：瓦特)

% 材料参数
m_shaft = 10; % 主轴材料 Wohler 曲线斜率
C_shaft = 9.77e70; % 主轴材料常数
m_tower = 10; % 塔架材料 Wohler 曲线斜率
C_tower = 9.77e70; % 塔架材料常数
sigma_b = 5e7; % 材料在拉伸断裂时的最大载荷值（Pa）

% 风机和空气参数定义
rho = 1.225; % 空气密度 kg/m^3
R = 72.5; % 风轮半径 m
A = pi * R^2; % 风轮扫掠面积 m^2

% 加载回归模型
load('mdl_Cp.mat', 'mdl_Cp'); % 加载功率系数 Cp 的回归模型
load('mdl_Ct.mat', 'mdl_Ct'); % 加载推力系数 Ct 的回归模型
% 加载特征均值和标准差，用于特征标准化
load('feature_mean_std.mat', 'feature_mean', 'feature_std'); % 包含 feature_mean 和 feature_std

%% 初始化每台风机的载荷历史（用于马尔可夫链模型）
% 定义马尔可夫链的状态数和转移概率矩阵
num_states_shaft = 10; % 主轴疲劳状态数
num_states_tower = 10; % 塔架疲劳状态数

% 加载状态转移矩阵
load('transition_matrix_shaft.mat'); % 主轴状态转移矩阵
load('transition_matrix_tower.mat'); % 塔架状态转移矩阵

% 初始状态分布
state_distribution_shaft = zeros(1, num_states_shaft);
state_distribution_shaft(1) = 1; % 初始时刻主轴完全处于第一个状态
state_distribution_tower = zeros(1, num_states_tower);
state_distribution_tower(1) = 1; % 初始时刻塔架完全处于第一个状态

% 定义每个状态对应的疲劳损伤增量（根据实际情况调整）
fatigue_increment_shaft = linspace(0, 2e-17, num_states_shaft);
fatigue_increment_tower = linspace(0, 1e-17, num_states_tower);

%% 定义目标函数权重
w_shaft = 0.5; % 主轴疲劳损伤的权重
w_tower = 0.5; % 塔架疲劳损伤的权重

%% 定义优化选项
% 粒子群优化选项设置
pso_options = optimoptions('particleswarm', ...
    'SwarmSize', 100, ...               % 粒子数
    'MaxIterations', 200, ...           % 最大迭代次数
    'FunctionTolerance', 1e-6, ...      % 函数容忍度
    'Display', 'off', ...               % 不显示迭代过程
    'UseParallel', false);              % 禁用并行计算（可根据需要启用）

%% 创建函数用于运行优化过程
% 定义函数 runOptimization
function [results] = runOptimization(useKalmanFilter, useARIMAPrediction, ...
    wind_speed_test, pitch_angle_test, omega_r_test, power_out_test, power_schedule_test, ...
    wind_speed_train, pitch_angle_train, omega_r_train, power_out_train, ...
    Pref_test, total_time, num_turbines_test_used, ...
    P_max, Delta_P_max, m_shaft, C_shaft, m_tower, C_tower, sigma_b, ...
    rho, R, A, mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    num_states_shaft, num_states_tower, transition_matrix_shaft, transition_matrix_tower, ...
    state_distribution_shaft, state_distribution_tower, ...
    fatigue_increment_shaft, fatigue_increment_tower, w_shaft, w_tower, pso_options)

    % 初始化风机数量
    num_turbines = num_turbines_test_used;

    % 初始化累积疲劳损伤
    cumulative_damage_shaft_opt = zeros(total_time, num_turbines); % 优化后主轴累积疲劳损伤
    cumulative_damage_tower_opt = zeros(total_time, num_turbines); % 优化后塔架累积疲劳损伤

    % 记录每一秒的功率分配
    power_reference_opt = zeros(total_time, num_turbines); % 优化后的功率参考值

    % 初始化累积疲劳损伤总和
    cumulative_damage_shaft_opt_totals = zeros(num_turbines, 1);
    cumulative_damage_tower_opt_totals = zeros(num_turbines, 1);

    %% 数据预处理
    if useKalmanFilter
        % 应用卡尔曼滤波减少测量噪声
        % 初始化滤波器参数
        kf_A = 1; % 状态转移矩阵
        kf_H = 1; % 测量矩阵
        kf_Q = 1e-5; % 过程噪声协方差
        kf_R = (0.1 * ones(total_time, num_turbines)).^2; % 测量噪声协方差（相对10%）

        % 初始化卡尔曼滤波器的估计误差协方差和状态
        kf_P = ones(num_turbines, 1);
        kf_x = zeros(num_turbines, 1);

        % 对测试数据应用卡尔曼滤波
        wind_speed_filtered_test = zeros(total_time, num_turbines);
        pitch_angle_filtered_test = zeros(total_time, num_turbines);
        omega_r_filtered_test = zeros(total_time, num_turbines);
        power_out_filtered_test = zeros(total_time, num_turbines);

        for t = 1:total_time
            for turbine = 1:num_turbines
                % 风速滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), wind_speed_test(t, turbine));
                wind_speed_filtered_test(t, turbine) = kf_x(turbine);

                % 桨距角滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), pitch_angle_test(t, turbine));
                pitch_angle_filtered_test(t, turbine) = kf_x(turbine);

                % 转速滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), omega_r_test(t, turbine));
                omega_r_filtered_test(t, turbine) = kf_x(turbine);

                % 输出功率滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), power_out_test(t, turbine));
                power_out_filtered_test(t, turbine) = kf_x(turbine);
            end
        end
    else
        % 不应用卡尔曼滤波，直接使用原始数据
        wind_speed_filtered_test = wind_speed_test;
        pitch_angle_filtered_test = pitch_angle_test;
        omega_r_filtered_test = omega_r_test;
        power_out_filtered_test = power_out_test;
    end

    %% ARIMA模型
    if useARIMAPrediction
        % 训练ARIMA模型
        num_turbines_ARIMA = num_turbines; % 对测试的风机进行预测

        % 初始化ARIMA模型存储
        arima_models_V = cell(num_turbines_ARIMA,1); % 风速
        arima_models_pitch = cell(num_turbines_ARIMA,1); % 桨距角
        arima_models_omega = cell(num_turbines_ARIMA,1); % 转速
        arima_models_Pout = cell(num_turbines_ARIMA,1); % 输出功率

        % 对每台风机训练ARIMA模型
        for turbine = 1:num_turbines_ARIMA
            % 风速
            ts_V_train = wind_speed_train(:, turbine);
            arima_models_V{turbine} = estimateARIMA(ts_V_train);

            % 桨距角
            ts_pitch_train = pitch_angle_train(:, turbine);
            arima_models_pitch{turbine} = estimateARIMA(ts_pitch_train);

            % 转速
            ts_omega_train = omega_r_train(:, turbine);
            arima_models_omega{turbine} = estimateARIMA(ts_omega_train);

            % 输出功率
            ts_Pout_train = power_out_train(:, turbine);
            arima_models_Pout{turbine} = estimateARIMA(ts_Pout_train);
        end
    end

    %% 模拟优化过程
    tic; % 开始计时

    % 最大延迟设置
    max_delay = 10; % 最大延迟（秒）

    % 初始化数据缓冲索引
    buffer_indices = ones(num_turbines,1); % 每台风机的缓冲区当前索引

    % 初始化数据缓冲区
    data_buffer_V = cell(max_delay, num_turbines); % 风速
    data_buffer_pitch = cell(max_delay, num_turbines); % 桨距角
    data_buffer_omega = cell(max_delay, num_turbines); % 转速
    data_buffer_Pout = cell(max_delay, num_turbines); % 输出功率

    % 初始化缓冲区，填充初始数据（使用测试数据的第一个时刻值）
    for d = 1:max_delay
        for turbine = 1:num_turbines
            data_buffer_V{d, turbine} = wind_speed_filtered_test(1, turbine);
            data_buffer_pitch{d, turbine} = pitch_angle_filtered_test(1, turbine);
            data_buffer_omega{d, turbine} = omega_r_filtered_test(1, turbine);
            data_buffer_Pout{d, turbine} = power_out_filtered_test(1, turbine);
        end
    end

    % 初始化变量
    wind_speed_predicted = zeros(total_time, num_turbines);
    pitch_angle_predicted = zeros(total_time, num_turbines);
    omega_r_predicted = zeros(total_time, num_turbines);
    power_out_predicted = zeros(total_time, num_turbines);

    % 遍历每一个时间步进行优化
    for t = 1:total_time
        %% 模拟随机通信延迟（1到10秒）
        current_delay = randi([1, max_delay], num_turbines, 1); % 每台风机随机延迟时间
        delayed_t = t - current_delay;

        for turbine = 1:num_turbines
            if delayed_t(turbine) > 0 && delayed_t(turbine) <= total_time
                % 获取延迟后的时刻数据
                curr_idx = mod(delayed_t(turbine)-1, max_delay) + 1;
                current_data_V = data_buffer_V{curr_idx, turbine};
                current_data_pitch = data_buffer_pitch{curr_idx, turbine};
                current_data_omega = data_buffer_omega{curr_idx, turbine};
                current_data_Pout = data_buffer_Pout{curr_idx, turbine};
            else
                % 如果数据因延迟不可用，使用预测模型填补数据
                if useARIMAPrediction && exist('arima_models_V', 'var')
                    if t >1
                        % 使用上一时刻的预测值
                        current_data_V = wind_speed_predicted(t-1, turbine);
                        current_data_pitch = pitch_angle_predicted(t-1, turbine);
                        current_data_omega = omega_r_predicted(t-1, turbine);
                        current_data_Pout = power_out_predicted(t-1, turbine);
                    else
                        % 初始数据
                        current_data_V = wind_speed_filtered_test(t, turbine);
                        current_data_pitch = pitch_angle_filtered_test(t, turbine);
                        current_data_omega = omega_r_filtered_test(t, turbine);
                        current_data_Pout = power_out_filtered_test(t, turbine);
                    end
                    % 使用ARIMA模型预测下一步的数据
                    % 风速预测
                    try
                        prediction_V = forecast(arima_models_V{turbine},1,'Y0',wind_speed_train(:, turbine));
                        wind_speed_predicted(t, turbine) = prediction_V;
                    catch
                        prediction_V = current_data_V; % 如果预测失败，使用当前值
                        wind_speed_predicted(t, turbine) = prediction_V;
                    end

                    % 桨距角预测
                    try
                        prediction_pitch = forecast(arima_models_pitch{turbine},1,'Y0',pitch_angle_train(:, turbine));
                        pitch_angle_predicted(t, turbine) = prediction_pitch;
                    catch
                        prediction_pitch = current_data_pitch;
                        pitch_angle_predicted(t, turbine) = prediction_pitch;
                    end

                    % 转速预测
                    try
                        prediction_omega = forecast(arima_models_omega{turbine},1,'Y0',omega_r_train(:, turbine));
                        omega_r_predicted(t, turbine) = prediction_omega;
                    catch
                        prediction_omega = current_data_omega;
                        omega_r_predicted(t, turbine) = prediction_omega;
                    end

                    % 输出功率预测
                    try
                        prediction_Pout = forecast(arima_models_Pout{turbine},1,'Y0',power_out_train(:, turbine));
                        power_out_predicted(t, turbine) = prediction_Pout;
                    catch
                        prediction_Pout = current_data_Pout;
                        power_out_predicted(t, turbine) = prediction_Pout;
                    end

                    % 更新当前数据为预测值
                    current_data_V = wind_speed_predicted(t, turbine);
                    current_data_pitch = pitch_angle_predicted(t, turbine);
                    current_data_omega = omega_r_predicted(t, turbine);
                    current_data_Pout = power_out_predicted(t, turbine);
                else
                    % 不使用ARIMA预测，直接使用上一时刻的数据或默认值
                    if t >1
                        current_data_V = wind_speed_filtered_test(t-1, turbine);
                        current_data_pitch = pitch_angle_filtered_test(t-1, turbine);
                        current_data_omega = omega_r_filtered_test(t-1, turbine);
                        current_data_Pout = power_out_filtered_test(t-1, turbine);
                    else
                        current_data_V = wind_speed_filtered_test(t, turbine);
                        current_data_pitch = pitch_angle_filtered_test(t, turbine);
                        current_data_omega = omega_r_filtered_test(t, turbine);
                        current_data_Pout = power_out_filtered_test(t, turbine);
                    end
                end
            end

            % 更新预测数据和缓冲区
            wind_speed_predicted(t, turbine) = current_data_V;
            pitch_angle_predicted(t, turbine) = current_data_pitch;
            omega_r_predicted(t, turbine) = current_data_omega;
            power_out_predicted(t, turbine) = current_data_Pout;

            % 添加到数据缓冲中
            buffer_idx = mod(t-1, max_delay) + 1;
            data_buffer_V{buffer_idx, turbine} = current_data_V;
            data_buffer_pitch{buffer_idx, turbine} = current_data_pitch;
            data_buffer_omega{buffer_idx, turbine} = current_data_omega;
            data_buffer_Pout{buffer_idx, turbine} = current_data_Pout;
        end

        %% 获取最新的可用数据（上一个时刻的数据）
        if t >1
            data_available_V = wind_speed_predicted(t-1, :);
            data_available_pitch = pitch_angle_predicted(t-1, :);
            data_available_omega = omega_r_predicted(t-1, :);
            data_available_Pout = power_out_predicted(t-1, :);
        else
            data_available_V = wind_speed_filtered_test(t, :);
            data_available_pitch = pitch_angle_filtered_test(t, :);
            data_available_omega = omega_r_filtered_test(t, :);
            data_available_Pout = power_out_filtered_test(t, :);
        end

        %% 当前时刻的总调度指令功率
        P_total = sum(Pref_test(t, :)); % 使用测试调度指令 Pref_test

        %% 平均分配功率
        P_avg = P_total / num_turbines;
        power_reference_avg = P_avg; % 平均分配的功率

        %% 获取当前时刻每台风机的风速和其他参数
        V_t = data_available_V';          % 风速 (num_turbines x 1)
        pitch_t = data_available_pitch';  % 桨距角 (num_turbines x 1)
        omega_t = data_available_omega';  % 转速 (num_turbines x 1)
        P_out_t = data_available_Pout';   % 实际输出功率 (num_turbines x 1)
        Pref_t = Pref_test(t, :)';        % 调度功率指令 (num_turbines x 1)

        %% 对测量数据进行卡尔曼滤波处理，减小噪声影响
        if useKalmanFilter
            V_filtered = zeros(num_turbines,1);
            pitch_filtered = zeros(num_turbines,1);
            omega_filtered = zeros(num_turbines,1);
            P_out_filtered = zeros(num_turbines,1);

            for turbine =1:num_turbines
                % 风速滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), V_t(turbine));
                V_filtered(turbine) = kf_x(turbine);

                % 桨距角滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), pitch_t(turbine));
                pitch_filtered(turbine) = kf_x(turbine);

                % 转速滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), omega_t(turbine));
                omega_filtered(turbine) = kf_x(turbine);

                % 输出功率滤波
                [kf_x(turbine), kf_P(turbine)] = kalmanFilter(kf_A, kf_H, kf_Q, kf_R(t, turbine), kf_x(turbine), kf_P(turbine), P_out_t(turbine));
                P_out_filtered(turbine) = kf_x(turbine);
            end
        else
            V_filtered = V_t;
            pitch_filtered = pitch_t;
            omega_filtered = omega_t;
            P_out_filtered = P_out_t;
        end

        %% 定义优化变量的不确定性范围（基于10%噪声）
        uncertainty_percentage = 0.1;
        V_uncertainty = uncertainty_percentage * V_filtered;
        pitch_uncertainty = uncertainty_percentage * pitch_filtered;
        omega_uncertainty = uncertainty_percentage * omega_filtered;
        P_out_uncertainty = uncertainty_percentage * P_out_filtered;

        %% 定义优化变量的上下界
        P_available = calculateAvailablePower(V_filtered); % 计算每台风机的最大可用功率
        ub = min(P_available, P_max * ones(num_turbines, 1)); % 上界
        lb = zeros(num_turbines, 1);                          % 下界

        %% 定义优化目标函数：最小化加权后的总体疲劳损伤，并添加惩罚项
        obj_fun = @(P) weightedSumFatigueDamageWithPenalty(P, V_filtered, pitch_filtered, omega_filtered, P_out_filtered, ...
            w_shaft, w_tower, sigma_b, m_shaft, C_shaft, m_tower, C_tower, ...
            mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
            fatigue_increment_shaft, fatigue_increment_tower, ...
            transition_matrix_shaft, transition_matrix_tower, ...
            state_distribution_shaft, state_distribution_tower, ...
            num_states_shaft, num_states_tower, ...
            Delta_P_max, P_avg, P_total, ...
            V_uncertainty, pitch_uncertainty, omega_uncertainty, P_out_uncertainty);

        %% 粒子群优化，维度为num_turbines
        [P_opt, ~] = particleswarm(obj_fun, num_turbines, lb, ub, pso_options);
        P_opt = P_opt(:); % 确保P_opt是列向量

        %% 记录优化后的功率分配
        power_reference_opt(t, :) = P_opt';

        %% 估算优化后的主轴扭矩和塔架推力
        [T_shaft_opt, F_tower_opt] = estimateLoads(P_opt, V_filtered, pitch_filtered, omega_filtered, P_out_filtered, ...
            mdl_Cp, mdl_Ct, feature_mean, feature_std);

        %% 更新疲劳损伤（使用马尔可夫链模型）
        Di_shaft_opt_scalar = state_distribution_shaft * fatigue_increment_shaft';
        Di_tower_opt_scalar = state_distribution_tower * fatigue_increment_tower';

        % 马尔可夫链模型权重等（根据实际情况调整）
        w_state_distribution_for_shaft = 0.6290;
        w_state_distribution_for_tower = 0.6413;
        w_shaft_fatigue = 0.204946;
        w_tower_fatigue = 0.192633;
        w_P_fatigue = 0.166139;

        % 将标量扩展为向量，广播到所有风机
        Di_shaft_opt = w_state_distribution_for_shaft * Di_shaft_opt_scalar * ones(num_turbines, 1) + ...
                       w_shaft_fatigue * T_shaft_opt + w_P_fatigue * P_opt;
        Di_tower_opt = w_state_distribution_for_tower * Di_tower_opt_scalar * ones(num_turbines, 1) + ...
                       w_tower_fatigue * F_tower_opt + w_P_fatigue * P_opt;

        %% 累积疲劳损伤
        cumulative_damage_shaft_opt_totals = cumulative_damage_shaft_opt_totals + Di_shaft_opt;
        cumulative_damage_tower_opt_totals = cumulative_damage_tower_opt_totals + Di_tower_opt;

        %% 记录累积疲劳损伤
        cumulative_damage_shaft_opt(t, :) = cumulative_damage_shaft_opt_totals';
        cumulative_damage_tower_opt(t, :) = cumulative_damage_tower_opt_totals';

        %% 更新马尔可夫链状态分布
        state_distribution_shaft = state_distribution_shaft * transition_matrix_shaft;
        state_distribution_tower = state_distribution_tower * transition_matrix_tower;

        %% 显示优化进度
        fprintf('已完成 %d 秒的优化计算。\n', t);
    end

    toc; % 结束计时

    %% 优化后总累积疲劳损伤
    total_damage_shaft_opt = sum(cumulative_damage_shaft_opt, 2);    % 优化后主轴总损伤 (total_time x 1)
    total_damage_tower_opt = sum(cumulative_damage_tower_opt, 2);    % 优化后塔架总损伤 (total_time x 1)

    % 分别计算加权后的总体疲劳损伤
    weighted_total_damage_opt = w_shaft * total_damage_shaft_opt + w_tower * total_damage_tower_opt; % (total_time x 1)

    % 保存结果
    results.power_reference_opt = power_reference_opt;
    results.total_damage_shaft_opt = total_damage_shaft_opt;
    results.total_damage_tower_opt = total_damage_tower_opt;
    results.weighted_total_damage_opt = weighted_total_damage_opt;
end

%% 辅助函数定义

% 卡尔曼滤波器函数
function [x_new, P_new] = kalmanFilter(A, H, Q, R, x_prev, P_prev, z)
    % 预测步骤
    x_pred = A * x_prev;
    P_pred = A * P_prev * A' + Q;

    % 更新步骤
    K = P_pred * H' / (H * P_pred * H' + R);
    x_new = x_pred + K * (z - H * x_pred);
    P_new = (1 - K * H) * P_pred;
end

% ARIMA模型训练函数
function model = estimateARIMA(time_series)
    % 自动选择ARIMA模型参数，使用Matlab的 estimate 函数
    % 这里使用简单的自回归移动平均模型
    % 需要使用 Econometrics Toolbox
    try
        model = estimate(arima('Constant',0,'D',1,'Seasonality',0,'MALags',1,'ARLags',1), time_series, 'Display', 'off');
    catch
        % 如果估计失败，使用默认模型
        model = arima('Constant',0,'D',1,'Seasonality',0,'MALags',1,'ARLags',1);
    end
end

% 优化目标函数：最小化加权后的总体疲劳损伤，并添加惩罚项
function weighted_damage = weightedSumFatigueDamageWithPenalty(P, V, pitch, omega_r, P_out, ...
    w_shaft, w_tower, sigma_b, m_shaft, C_shaft, m_tower, C_tower, ...
    mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    fatigue_increment_shaft, fatigue_increment_tower, ...
    transition_matrix_shaft, transition_matrix_tower, ...
    state_distribution_shaft, state_distribution_tower, ...
    num_states_shaft, num_states_tower, ...
    Delta_P_max, P_avg, P_total, ...
    V_uncertainty, pitch_uncertainty, omega_uncertainty, P_out_uncertainty)

    % 确保P是列向量
    P = P(:);

    num_turbines = length(P);

    % 计算总功率分配误差（调度指令功率与优化功率之差）
    power_diff_total = sum(P) - P_total;
    penalty_power_sum = 1e6 * abs(power_diff_total); % 大的惩罚

    % 计算功率波动约束的违背程度
    power_fluctuation = abs(P - P_avg) - Delta_P_max;
    penalty_power_fluctuation = 1e6 * sum(power_fluctuation(power_fluctuation > 0));

    % 估算应力/载荷
    [T_shaft, F_tower] = estimateLoads(P, V, pitch, omega_r, P_out, ...
        mdl_Cp, mdl_Ct, feature_mean, feature_std);

    % 初始化损伤增量
    Di_shaft_opt = zeros(num_turbines,1);
    Di_tower_opt = zeros(num_turbines,1);

    % 计算每台风机的疲劳损伤增量
    Di_shaft_opt = state_distribution_shaft * fatigue_increment_shaft';
    Di_tower_opt = state_distribution_tower * fatigue_increment_tower';

    % 计算总疲劳损伤
    total_damage_shaft = sum(Di_shaft_opt);
    total_damage_tower = sum(Di_tower_opt);

    % 计算加权后的总体疲劳损伤
    weighted_damage = w_shaft * total_damage_shaft + w_tower * total_damage_tower + ...
                      penalty_power_sum + penalty_power_fluctuation;
end

% 估算应力/载荷（主轴扭矩和塔架推力），使用回归模型
function [T_shaft, F_tower] = estimateLoads(P, V, pitch, omega_r, P_out, ...
    mdl_Cp, mdl_Ct, feature_mean, feature_std)
    % 计算叶尖速比 λ
    lambda = (omega_r .* 72.5) ./ V; % R = 72.5 m

    % 计算功率差值
    power_diff = P - P_out;

    % 构建特征矩阵，包括叶尖速比、桨距角、功率和功率差
    data = [lambda, pitch, P, power_diff];

    % 标准化特征矩阵
    features_standardized = (data - feature_mean) ./ feature_std;

    % 使用回归模型预测功率系数 Cp 和推力系数 Ct
    Cp_predicted = predict(mdl_Cp, features_standardized);
    Ct_predicted = predict(mdl_Ct, features_standardized);

    % 限制 Cp 和 Ct 的值在合理范围内 [0, 1]
    Cp_predicted(Cp_predicted < 0) = 0;
    Cp_predicted(Cp_predicted > 1) = 1;
    Ct_predicted(Ct_predicted < 0) = 0;
    Ct_predicted(Ct_predicted > 1) = 1;

    % 估算主轴扭矩 T_shaft
    rho = 1.225; % 空气密度 kg/m^3
    R = 72.5; % 风轮半径 m
    A = pi * R^2; % 扫风面积 m^2
    P_estimated = 0.5 * rho * A .* Cp_predicted .* V .^ 3; % 估算功率
    T_shaft = P_estimated ./ omega_r; % 通过功率和转速计算扭矩
    % 处理无效值（NaN 或 Inf）
    T_shaft(isnan(T_shaft) | isinf(T_shaft)) = 0;

    % 估算塔架推力 F_tower
    F_tower = 0.5 * rho * A .* Ct_predicted .* V .^ 2;
end

% 计算风机的最大可用功率
function P_avail = calculateAvailablePower(V)
    rho = 1.225; % 空气密度 kg/m^3
    R = 72.5; % 风轮半径 m
    A = pi * R^2; % 风轮扫掠面积 m^2
    Cp_max = 0.48; % 最大功率系数（根据实际情况调整）
    P_rated = 5e6; % 额定功率 W
    V_rated = 11.2; % 额定风速 m/s

    % 计算未限幅的功率
    P_raw = 0.5 * rho * A * Cp_max .* V .^ 3; % P = 0.5 * ρ * A * Cp * V^3
    % 限制功率不超过额定功率
    P_avail = min(P_raw, P_rated * ones(size(V)));

    % 当风速超过额定风速时，功率保持在额定功率不再增加
    P_avail(V > V_rated) = P_rated;
end

%% 运行不同的实验
% 运行不同的实验
results_baseline = runOptimization(false, false, ...
    wind_speed_test, pitch_angle_test, omega_r_test, power_out_test, power_schedule_test, ...
    wind_speed_train, pitch_angle_train, omega_r_train, power_out_train, ...
    Pref_test, total_time, num_turbines_test_used, ...
    P_max, Delta_P_max, m_shaft, C_shaft, m_tower, C_tower, sigma_b, ...
    rho, R, A, mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    num_states_shaft, num_states_tower, transition_matrix_shaft, transition_matrix_tower, ...
    state_distribution_shaft, state_distribution_tower, ...
    fatigue_increment_shaft, fatigue_increment_tower, w_shaft, w_tower, pso_options);

results_kalman = runOptimization(true, false, ...
    wind_speed_test, pitch_angle_test, omega_r_test, power_out_test, power_schedule_test, ...
    wind_speed_train, pitch_angle_train, omega_r_train, power_out_train, ...
    Pref_test, total_time, num_turbines_test_used, ...
    P_max, Delta_P_max, m_shaft, C_shaft, m_tower, C_tower, sigma_b, ...
    rho, R, A, mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    num_states_shaft, num_states_tower, transition_matrix_shaft, transition_matrix_tower, ...
    state_distribution_shaft, state_distribution_tower, ...
    fatigue_increment_shaft, fatigue_increment_tower, w_shaft, w_tower, pso_options);

results_arima = runOptimization(false, true, ...
    wind_speed_test, pitch_angle_test, omega_r_test, power_out_test, power_schedule_test, ...
    wind_speed_train, pitch_angle_train, omega_r_train, power_out_train, ...
    Pref_test, total_time, num_turbines_test_used, ...
    P_max, Delta_P_max, m_shaft, C_shaft, m_tower, C_tower, sigma_b, ...
    rho, R, A, mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    num_states_shaft, num_states_tower, transition_matrix_shaft, transition_matrix_tower, ...
    state_distribution_shaft, state_distribution_tower, ...
    fatigue_increment_shaft, fatigue_increment_tower, w_shaft, w_tower, pso_options);

results_full = runOptimization(true, true, ...
    wind_speed_test, pitch_angle_test, omega_r_test, power_out_test, power_schedule_test, ...
    wind_speed_train, pitch_angle_train, omega_r_train, power_out_train, ...
    Pref_test, total_time, num_turbines_test_used, ...
    P_max, Delta_P_max, m_shaft, C_shaft, m_tower, C_tower, sigma_b, ...
    rho, R, A, mdl_Cp, mdl_Ct, feature_mean, feature_std, ...
    num_states_shaft, num_states_tower, transition_matrix_shaft, transition_matrix_tower, ...
    state_distribution_shaft, state_distribution_tower, ...
    fatigue_increment_shaft, fatigue_increment_tower, w_shaft, w_tower, pso_options);

%% 绘制累积疲劳损伤对比图
figure;
plot(1:total_time, results_baseline.weighted_total_damage_opt, 'k-', 'LineWidth', 1.5, 'DisplayName', '无滤波和预测');
hold on;
plot(1:total_time, results_kalman.weighted_total_damage_opt, 'b--', 'LineWidth', 1.5, 'DisplayName', '仅卡尔曼滤波');
plot(1:total_time, results_arima.weighted_total_damage_opt, 'g-.', 'LineWidth', 1.5, 'DisplayName', '仅ARIMA预测');
plot(1:total_time, results_full.weighted_total_damage_opt, 'r-', 'LineWidth', 2, 'DisplayName', '滤波和预测结合');
xlabel('时间 (s)');
ylabel('加权总体累积疲劳损伤');
legend;
title('不同模型下加权总体累积疲劳损伤对比');
grid on;

%% 绘制功率分配方差对比图
variance_baseline = var(results_baseline.power_reference_opt, 0, 2) / 1e12;
variance_kalman = var(results_kalman.power_reference_opt, 0, 2) / 1e12;
variance_arima = var(results_arima.power_reference_opt, 0, 2) / 1e12;
variance_full = var(results_full.power_reference_opt, 0, 2) / 1e12;

figure;
plot(1:total_time, variance_baseline, 'k-', 'LineWidth', 1.5, 'DisplayName', '无滤波和预测');
hold on;
plot(1:total_time, variance_kalman, 'b--', 'LineWidth', 1.5, 'DisplayName', '仅卡尔曼滤波');
plot(1:total_time, variance_arima, 'g-.', 'LineWidth', 1.5, 'DisplayName', '仅ARIMA预测');
plot(1:total_time, variance_full, 'r-', 'LineWidth', 2, 'DisplayName', '滤波和预测结合');
xlabel('时间 (s)');
ylabel('功率分配方差 (MW^2)');
legend;
title('不同模型下功率分配方差对比');
grid on;