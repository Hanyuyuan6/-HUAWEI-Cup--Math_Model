clc; clear; close all; % 清除命令窗口，清除工作区变量，关闭所有图形窗口

%% 加载数据
% 从数据文件中加载噪声和延迟作用下风电机组数据 (附件3: 训练数据)
load('附件3-噪声和延迟作用下的采集数据.mat'); % 附件3-噪声和延迟作用下的采集数据为 data_TS_WF_noise

% 加载附件4的数据：额外的十台风机的 300s 测量数据, 作为测试集检验鲁棒模型
filename_test = '附件4-噪声和延迟作用下的采集数据.xlsx';  % Excel 文件名
startRow = 10;     % 开始读取的行
numRows = 100;      % 读取的数据行数
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

% 遍历每台风机，提取相应的数据（取前20秒）
for turbine = 1:num_turbines_train
    power_schedule_train(:, turbine) = wind_farm_train{turbine}.inputs(1:total_time,1); % 提取调度指令数据 Pref
    wind_speed_train(:, turbine) = wind_farm_train{turbine}.inputs(1:total_time,2); % 提取风速数据
    pitch_angle_train(:, turbine) = wind_farm_train{turbine}.states(1:total_time,1); % 提取桨距角数据
    omega_r_train(:, turbine) = wind_farm_train{turbine}.states(1:total_time,2); % 提取转速数据
    power_out_train(:, turbine) = wind_farm_train{turbine}.outputs(1:total_time,3); % 提取实际输出功率数据
end

% 将测试集中10台风机的数据直接使用
power_schedule_test = Pref_test; % 调度指令 Pref (20x10)
wind_speed_test = Vwin_test; % 风速 (20x10)
pitch_angle_test = Ft_test; % 桨距角 Ft (20x10)
omega_r_test = wgenM_test; % 转速 (20x10)
power_out_test = Pout_test; % 输出功率 (20x10)

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

%% 数据预处理
% 应用卡尔曼滤波减少测量噪声
% 初始化滤波器参数
kf_A = 1; % 状态转移矩阵
kf_H = 1; % 测量矩阵
kf_Q = 1e-5; % 过程噪声协方差
kf_R = (0.1 * ones(total_time, num_turbines_test_used)).^2; % 测量噪声协方差（相对10%）

% 初始化卡尔曼滤波器的估计误差协方差
kf_P = 1 * ones(num_turbines_test_used, 1);

% 初始化估计状态
kf_x = zeros(num_turbines_test_used, 1);

% 对测试数据应用卡尔曼滤波
wind_speed_filtered_test = zeros(total_time, num_turbines_test_used);
pitch_angle_filtered_test = zeros(total_time, num_turbines_test_used);
omega_r_filtered_test = zeros(total_time, num_turbines_test_used);
power_out_filtered_test = zeros(total_time, num_turbines_test_used);

for t = 1:total_time
    for turbine = 1:num_turbines_test_used
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

%% 训练时间序列预测模型（ARIMA）
% 使用附件3的数据（训练数据）训练ARIMA模型
% 对于测试的10台风机，从训练数据中选择相应的风机用于训练
% 假设前10台风机用于测试，可以选择训练对应10台风机的ARIMA模型

num_turbines_ARIMA = num_turbines_test_used; % 对测试的10台风机进行预测

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

%% 初始化参数
% 最大延迟设置
max_delay = 10; % 最大延迟（秒）

% 初始化数据缓冲索引
buffer_indices = ones(num_turbines_ARIMA,1); % 每台风机的缓冲区当前索引

% 初始化数据缓冲区
data_buffer_V = cell(max_delay, num_turbines_ARIMA); % 风速
data_buffer_pitch = cell(max_delay, num_turbines_ARIMA); % 桨距角
data_buffer_omega = cell(max_delay, num_turbines_ARIMA); % 转速
data_buffer_Pout = cell(max_delay, num_turbines_ARIMA); % 输出功率

% 初始化缓冲区，填充初始数据（使用测试数据的第一个时刻值）
for d = 1:max_delay
    for turbine = 1:num_turbines_ARIMA
        data_buffer_V{d, turbine} = wind_speed_filtered_test(1, turbine);
        data_buffer_pitch{d, turbine} = pitch_angle_filtered_test(1, turbine);
        data_buffer_omega{d, turbine} = omega_r_filtered_test(1, turbine);
        data_buffer_Pout{d, turbine} = power_out_filtered_test(1, turbine);
    end
end

%% 粒子群优化选项设置
pso_options = optimoptions('particleswarm', ...
    'SwarmSize', 100, ...               % 粒子数
    'MaxIterations', 200, ...           % 最大迭代次数
    'FunctionTolerance', 1e-6, ...      % 函数容忍度
    'Display', 'off', ...               % 不显示迭代过程
    'UseParallel', false);              % 禁用并行计算（可根据需要启用）

%% 初始化累积疲劳损伤
cumulative_damage_shaft_opt = zeros(total_time, num_turbines_ARIMA); % 优化后主轴累积疲劳损伤
cumulative_damage_tower_opt = zeros(total_time, num_turbines_ARIMA); % 优化后塔架累积疲劳损伤
cumulative_damage_shaft_avg = zeros(total_time, num_turbines_ARIMA); % 平均分配主轴累积疲劳损伤
cumulative_damage_tower_avg = zeros(total_time, num_turbines_ARIMA); % 平均分配塔架累积疲劳损伤

% 记录每一秒的功率分配
power_reference_opt = zeros(total_time, num_turbines_ARIMA); % 优化后的功率参考值
power_reference_avg = zeros(total_time, num_turbines_ARIMA); % 平均分配的功率参考值

%% 优化求解过程
tic; % 开始计时

% 初始化累积疲劳损伤总和
cumulative_damage_shaft_opt_totals = zeros(num_turbines_ARIMA, 1);
cumulative_damage_tower_opt_totals = zeros(num_turbines_ARIMA, 1);
cumulative_damage_shaft_avg_totals = zeros(num_turbines_ARIMA, 1);
cumulative_damage_tower_avg_totals = zeros(num_turbines_ARIMA, 1);

% 遍历每一个时间步进行优化
for t = 1:total_time
    %% 模拟随机通信延迟（1到10秒）
    current_delay = randi([1, max_delay]); % 随机延迟时间
    delayed_t = t - current_delay;
    
    for turbine = 1:num_turbines_ARIMA
        if delayed_t > 0 && delayed_t <= total_time
            % 获取延迟后的时刻数据
            current_data_V = data_buffer_V{mod(delayed_t, max_delay)+1, turbine};
            current_data_pitch = data_buffer_pitch{mod(delayed_t, max_delay)+1, turbine};
            current_data_omega = data_buffer_omega{mod(delayed_t, max_delay)+1, turbine};
            current_data_Pout = data_buffer_Pout{mod(delayed_t, max_delay)+1, turbine};
        else
            % 如果数据因延迟不可用，使用预测模型填补数据
            if t >1
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
        end
        
        % 更新预测数据和缓冲区
        wind_speed_predicted(t, turbine) = current_data_V;
        pitch_angle_predicted(t, turbine) = current_data_pitch;
        omega_r_predicted(t, turbine) = current_data_omega;
        power_out_predicted(t, turbine) = current_data_Pout;
        
        % 添加到数据缓冲中
        buffer_idx = mod(t, max_delay) + 1;
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
    P_avg = P_total / num_turbines_ARIMA;
    power_reference_avg(t, :) = P_avg; % 记录平均分配的功率
    
    %% 获取当前时刻每台风机的风速和其他参数
    V_t = data_available_V';          % 风速 (10x1)
    pitch_t = data_available_pitch';  % 桨距角 (10x1)
    omega_t = data_available_omega';  % 转速 (10x1)
    P_out_t = data_available_Pout';   % 实际输出功率 (10x1)
    Pref_t = Pref_test(t, :)';        % 调度功率指令 (10x1)
    
    %% 对测量数据进行卡尔曼滤波处理，减小噪声影响
    [V_filtered, pitch_filtered, omega_filtered, P_out_filtered] = deal(zeros(num_turbines_ARIMA,1));
    for turbine =1:num_turbines_ARIMA
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
    
    %% 定义优化变量的不确定性范围（基于10%噪声）
    uncertainty_percentage = 0.1;
    V_uncertainty = uncertainty_percentage * V_filtered;
    pitch_uncertainty = uncertainty_percentage * pitch_filtered;
    omega_uncertainty = uncertainty_percentage * omega_filtered;
    P_out_uncertainty = uncertainty_percentage * P_out_filtered;
    
    %% 定义优化变量的上下界
    P_available = calculateAvailablePower(V_filtered); % 计算每台风机的最大可用功率
    ub = min(P_available, P_max * ones(num_turbines_ARIMA, 1)); % 上界
    lb = zeros(num_turbines_ARIMA, 1);                          % 下界
    
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
    
    %% 粒子群优化，维度为num_turbines_ARIMA
    [P_opt, ~] = particleswarm(obj_fun, num_turbines_ARIMA, lb, ub, pso_options);
    P_opt = P_opt(:); % 确保P_opt是列向量
    
    %% 记录优化后的功率分配
    power_reference_opt(t, :) = P_opt';
    
    %% 估算优化后的主轴扭矩和塔架推力
    [T_shaft_opt, F_tower_opt] = estimateLoads(P_opt, V_filtered, pitch_filtered, omega_filtered, P_out_filtered, ...
        mdl_Cp, mdl_Ct, feature_mean, feature_std);
    
    %% 估算平均分配的主轴扭矩和塔架推力
    [T_shaft_avg, F_tower_avg] = estimateLoads(P_avg * ones(num_turbines_ARIMA,1), V_filtered, pitch_filtered, omega_filtered, P_out_filtered, ...
        mdl_Cp, mdl_Ct, feature_mean, feature_std);
    
    %% 更新疲劳损伤（使用马尔可夫链模型）

    Di_shaft_opt_scalar = state_distribution_shaft * fatigue_increment_shaft';
    Di_tower_opt_scalar = state_distribution_tower * fatigue_increment_tower';
    Di_shaft_avg_scalar = state_distribution_shaft * fatigue_increment_shaft';
    Di_tower_avg_scalar = state_distribution_tower * fatigue_increment_tower';
    
    % 马尔可夫链模型权重等
    w_state_distribution_for_shaft = 0.6290;
    w_state_distribution_for_tower = 0.6413;
    w_shaft_fatigue = 0.204946;
    w_tower_fatigue = 0.192633;
    w_P_fatigue = 0.166139;
    
    % 将标量扩展为向量，广播到所有风机
    Di_shaft_opt = w_state_distribution_for_shaft * Di_shaft_opt_scalar * ones(num_turbines_ARIMA, 1) + ...
                   w_shaft_fatigue * T_shaft_opt + w_P_fatigue * P_opt;
    Di_tower_opt = w_state_distribution_for_tower * Di_tower_opt_scalar * ones(num_turbines_ARIMA, 1) + ...
                   w_tower_fatigue * F_tower_opt + w_P_fatigue * P_opt;
    Di_shaft_avg = w_state_distribution_for_shaft * Di_shaft_avg_scalar * ones(num_turbines_ARIMA, 1) + ...
                   w_shaft_fatigue * T_shaft_avg + w_P_fatigue * P_avg;
    Di_tower_avg = w_state_distribution_for_tower * Di_tower_avg_scalar * ones(num_turbines_ARIMA, 1) + ...
                   w_tower_fatigue * F_tower_avg + w_P_fatigue * P_avg;
    
    %% 累积疲劳损伤
    cumulative_damage_shaft_opt_totals = cumulative_damage_shaft_opt_totals + Di_shaft_opt;
    cumulative_damage_tower_opt_totals = cumulative_damage_tower_opt_totals + Di_tower_opt;
    cumulative_damage_shaft_avg_totals = cumulative_damage_shaft_avg_totals + Di_shaft_avg;
    cumulative_damage_tower_avg_totals = cumulative_damage_tower_avg_totals + Di_tower_avg;
    
    %% 记录累积疲劳损伤
    cumulative_damage_shaft_opt(t, :) = cumulative_damage_shaft_opt_totals';
    cumulative_damage_tower_opt(t, :) = cumulative_damage_tower_opt_totals';
    cumulative_damage_shaft_avg(t, :) = cumulative_damage_shaft_avg_totals';
    cumulative_damage_tower_avg(t, :) = cumulative_damage_tower_avg_totals';
    
    %% 更新马尔可夫链状态分布
    state_distribution_shaft = state_distribution_shaft * transition_matrix_shaft;
    state_distribution_tower = state_distribution_tower * transition_matrix_tower;
    
    %% 显示优化进度
    fprintf('已完成 %d 秒的优化计算。\n', t);
end

toc; % 结束计时

%% 结果可视化

% 定义视频文件的完整路径
videoFileName = 'question_4_优化功率分配动态展示.avi';

% 创建 VideoWriter 对象
v = VideoWriter(videoFileName, 'Uncompressed AVI'); % 你可以选择其他格式，如'MPEG-4'
v.FrameRate = 4; % 设置帧率，可以根据需要调整
open(v); % 打开视频文件准备写入

% 创建一个新的图形窗口
figure('Name', '功率分配动态展示', 'NumberTitle', 'off');

% 绘制功率分配的动态条形图
for t = 1:total_time
    % 绘制条形图
    bar(power_reference_opt(t, :) / 1e6, 'FaceColor', [0.2 0.6 0.8]); % 将功率转换为MW显示
    xlabel('风机编号');
    ylabel('功率参考值 (MW)');
    title(sprintf('第 %d 秒功率分配优化结果', t));
    ylim([0, P_max / 1e6 + 1]); % 设置y轴范围
    
    % 捕获当前图形并写入视频
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    % 更新图形
    drawnow;
    
    % 控制动画播放速度
    pause(0.01); % 设置为0.01秒，加快视频生成速度
end

% 关闭视频文件
close(v);

% 关闭图形窗口
close(gcf);

% 优化后总累积疲劳损伤
total_damage_shaft_opt = sum(cumulative_damage_shaft_opt, 2);    % 优化后主轴总损伤 (20x1)
total_damage_tower_opt = sum(cumulative_damage_tower_opt, 2);    % 优化后塔架总损伤 (20x1)
total_damage_shaft_avg = sum(cumulative_damage_shaft_avg, 2);    % 平均分配主轴总损伤 (20x1)
total_damage_tower_avg = sum(cumulative_damage_tower_avg, 2);    % 平均分配塔架总损伤 (20x1)

% 分别计算加权后的总体疲劳损伤
weighted_total_damage_opt = w_shaft * total_damage_shaft_opt + w_tower * total_damage_tower_opt; % (20x1)
weighted_total_damage_avg = w_shaft * total_damage_shaft_avg + w_tower * total_damage_tower_avg; % (20x1)

% 绘制加权后的总体疲劳损伤对比图
figure('Name', '优化前后加权总体累积疲劳损伤对比', 'NumberTitle', 'off');
plot(1:total_time, weighted_total_damage_opt, 'r-', 'LineWidth', 1.5); hold on; % 优化后加权总损伤
plot(1:total_time, weighted_total_damage_avg, 'b--', 'LineWidth', 1.5); % 平均分配加权总损伤
xlabel('时间 (s)');
ylabel('加权总体累积疲劳损伤');
legend({'加权总损伤（优化后）', '加权总损伤（平均分配）'}, 'Location', 'best');
title('优化前后加权总体累积疲劳损伤对比');
grid on;

% 优化前后功率参考值的方差对比
variance_opt = var(power_reference_opt, 0, 2); % 优化后每秒功率分配的方差 (20x1)
variance_avg = var(power_reference_avg, 0, 2); % 平均分配时功率分配的方差 (20x1)
variance_opt = variance_opt / 1e12;% 优化后方差 (转换为MW²)
variance_avg = variance_avg / 1e12;
% 绘制功率分配方差对比图
figure('Name', '优化前后功率分配方差对比', 'NumberTitle', 'off');
plot(1:total_time, variance_opt, 'g-', 'LineWidth', 1.5); hold on; 
plot(1:total_time, variance_avg, 'k--', 'LineWidth', 1.5); % 平均分配方差
xlabel('时间 (s)');
ylabel('功率分配方差 (MW^2)');
legend({'优化后', '平均分配'}, 'Location', 'best');
title('优化前后功率分配方差对比');
grid on;

% 选择5个有代表性的风机，展示累积疲劳损伤的增长过程
selected_turbines = [2, 3, 5, 7, 10]; % 选择的风机编号（1到10）

% 绘制优化后选定风机的累积疲劳损伤增长过程
figure('Name', '选定风机的累积疲劳损伤增长过程', 'NumberTitle', 'off');
single_total_damage_opt_total = [];
single_total_damage_avg_total = [];

for i = 1:length(selected_turbines)
    turbine = selected_turbines(i);
    single_total_damage_opt = w_shaft * cumulative_damage_shaft_opt(:, turbine) + w_tower * cumulative_damage_tower_opt(:, turbine);
    single_total_damage_avg = w_shaft * cumulative_damage_shaft_avg(:, turbine) + w_tower * cumulative_damage_tower_avg(:, turbine);
    single_total_damage_opt_total = [single_total_damage_opt_total,single_total_damage_opt];
    single_total_damage_avg_total = [single_total_damage_avg_total,single_total_damage_avg];
    subplot(length(selected_turbines), 1, i); % 创建子图
    plot(1:total_time, single_total_damage_opt, 'r-', 'LineWidth',1.5); hold on; % 优化后加权总损伤
    plot(1:total_time, single_total_damage_avg, 'b--', 'LineWidth',1.5); % 平均分配加权总损伤
    grid on;
    xlabel('时间 (s)');
    ylabel('累积疲劳损伤');
    legend({'优化后', '平均分配'}, 'Location', 'best');
    title(['风机 ', num2str(turbine), ' 累积疲劳损伤增长']);
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

% ARIMA模型预测函数
function prediction = forecast(model, steps, varargin)
    % 预测未来 steps 时刻的值，基于模型
    % 输入：
    %   model: 训练好的ARIMA模型
    %   steps: 预测的步骤数
    %   varargin: 'Y0', 最后一时刻的观测值
    if nargin > 2
        Y0 = varargin{2}; % varargin{1}是参数名称，如'Y0'
    else
        Y0 = [];
    end
    try
        [YF, ~] = forecast(model, steps, 'Y0', Y0);
        prediction = YF(end);
    catch
        % 如果预测失败，返回最后一个已知值
        if isempty(Y0)
            prediction = 0; % 默认值
        else
            prediction = Y0(end);
        end
    end
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