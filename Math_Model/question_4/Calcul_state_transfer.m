clc;            % 清空命令行窗口
clear;          % 清除工作区中的所有变量
close all;      % 关闭所有打开的图形窗口

%% 材料参数
m_shaft = 10;          % 主轴材料 Wohler 曲线斜率
C_shaft = 9.77e70;     % 主轴材料常数
m_tower = 10;          % 塔架材料 Wohler 曲线斜率
C_tower = 9.77e70;     % 塔架材料常数
sigma_b = 50e6;        % 材料在拉伸断裂时的最大载荷值（Pa）

%% 风机数量和时间步长
num_turbines = 100;    % 风机数量
total_time = 100;      % 总时间步长（秒）

%% 读取疲劳评估数据
filename = '附件1-疲劳评估数据.xls';  % Excel 文件名

% 读取主轴扭矩和塔架推力数据
[time, shaft_load, theoretical_shaft_LN, theoretical_shaft_Damage] = readLoadData(filename, '主轴扭矩', total_time, num_turbines);
[~, tower_load, theoretical_tower_LN, theoretical_tower_Damage] = readLoadData(filename, '塔架推力', total_time, num_turbines);

%% 使用简化模型计算主轴和塔架的累积疲劳损伤
% 在问题一答案中使用了雨流计数法和 Goodman 修正进行计算。由于实时计算要求，我们这里采用简化的方法。

% 简化的疲劳损伤计算（示例，仅供参考）
% 假设在每个时间步，疲劳损伤增量与载荷的幅值成正比
Damage_shaft_time = cumsum(abs(shaft_load).^(m_shaft), 1) * 1e-20;  % 示例系数
Damage_tower_time = cumsum(abs(tower_load).^(m_tower), 1) * 1e-20;  % 示例系数

%% 定义疲劳状态
num_states = 10;  % 将疲劳损伤程度划分为 10 个状态
% 获取主轴和塔架的累积疲劳损伤的最大值和最小值
max_damage_shaft = max(Damage_shaft_time(end, :));
min_damage_shaft = min(Damage_shaft_time(1, :));

max_damage_tower = max(Damage_tower_time(end, :));
min_damage_tower = min(Damage_tower_time(1, :));

% 定义状态边界（根据累积疲劳损伤值划分）
damage_levels_shaft = linspace(min_damage_shaft, max_damage_shaft, num_states + 1);
damage_levels_tower = linspace(min_damage_tower, max_damage_tower, num_states + 1);

%% 计算状态转移矩阵
% 初始化转移计数矩阵
transition_counts_shaft = zeros(num_states, num_states);
transition_counts_tower = zeros(num_states, num_states);

% 统计主轴的状态转移
for turbine = 1:num_turbines
    damage_series = Damage_shaft_time(:, turbine);  % 该风机的累积疲劳损伤时间序列
    % 将损伤值映射到状态
    [~, state_indices] = histc(damage_series, damage_levels_shaft);
    state_indices(state_indices == 0) = 1;  % 修正状态索引
    state_indices(state_indices > num_states) = num_states;
    % 统计转移次数
    for t = 1:length(state_indices) - 1
        i = state_indices(t);
        j = state_indices(t + 1);
        transition_counts_shaft(i, j) = transition_counts_shaft(i, j) + 1;
    end
end

% 统计塔架的状态转移
for turbine = 1:num_turbines
    damage_series = Damage_tower_time(:, turbine);  % 该风机的累积疲劳损伤时间序列
    % 将损伤值映射到状态
    [~, state_indices] = histc(damage_series, damage_levels_tower);
    state_indices(state_indices == 0) = 1;  % 修正状态索引
    state_indices(state_indices > num_states) = num_states;
    % 统计转移次数
    for t = 1:length(state_indices) - 1
        i = state_indices(t);
        j = state_indices(t + 1);
        transition_counts_tower(i, j) = transition_counts_tower(i, j) + 1;
    end
end

% 计算转移概率矩阵
transition_matrix_shaft = zeros(num_states, num_states);
transition_matrix_tower = zeros(num_states, num_states);

for i = 1:num_states
    total_transitions_shaft = sum(transition_counts_shaft(i, :));
    if total_transitions_shaft > 0
        transition_matrix_shaft(i, :) = transition_counts_shaft(i, :) / total_transitions_shaft;
    end
    total_transitions_tower = sum(transition_counts_tower(i, :));
    if total_transitions_tower > 0
        transition_matrix_tower(i, :) = transition_counts_tower(i, :) / total_transitions_tower;
    end
end

%% 显示状态转移矩阵
disp('主轴状态转移矩阵：');
disp(transition_matrix_shaft);

disp('塔架状态转移矩阵：');
disp(transition_matrix_tower);

%% 可视化状态转移矩阵
figure('Name', '主轴状态转移矩阵', 'NumberTitle', 'off');
imagesc(transition_matrix_shaft);
colorbar;
title('主轴状态转移矩阵');
xlabel('下一状态');
ylabel('当前状态');

figure('Name', '塔架状态转移矩阵', 'NumberTitle', 'off');
imagesc(transition_matrix_tower);
colorbar;
title('塔架状态转移矩阵');
xlabel('下一状态');
ylabel('当前状态');

%% 辅助函数定义

% 读取疲劳评估数据函数
function [time, load, theoretical_LN, theoretical_Damage] = readLoadData(filename, sheetName, total_time, num_turbines)
    % 读取 Excel 文件中的数据
    full_data = readmatrix(filename, 'Sheet', sheetName);
    time = full_data(1:total_time, 1);  % 提取时间列
    load = full_data(1:total_time, 2:num_turbines+1);  % 载荷数据 (100秒 × 100风机)
    theoretical_LN = full_data(total_time+1, 2:num_turbines+1);      % 理论等效疲劳载荷
    theoretical_Damage = full_data(total_time+2, 2:num_turbines+1);  % 理论累计疲劳损伤值
end