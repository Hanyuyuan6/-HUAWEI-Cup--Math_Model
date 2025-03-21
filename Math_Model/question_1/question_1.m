clc;            % 清空命令行窗口
clear;          % 清除工作区中的所有变量
close all;      % 关闭所有打开的图形窗口

%% 材料参数
m_shaft = 10;          % 主轴材料 Wohler 曲线斜率
C_shaft = 9.77e70;     % 主轴材料常数
m_tower = 10;          % 塔架材料 Wohler 曲线斜率
C_tower = 9.77e70;     % 塔架材料常数
sigma_b = 50000000;    % 材料在拉伸断裂时的最大载荷值

%% 风机数量和时间步长
num_turbines = 100;      % 风机数量
total_time = 100;        % 总时间步长（秒）

%% 读取疲劳评估数据
filename = '附件1-疲劳评估数据.xls';  

% 读取主轴扭矩和塔架推力数据
[time, shaft_load, theoretical_shaft_LN, theoretical_shaft_Damage] = readLoadData(filename, '主轴扭矩', total_time, num_turbines);
[~, tower_load, theoretical_tower_LN, theoretical_tower_Damage] = readLoadData(filename, '塔架推力', total_time, num_turbines);

%% 使用雨流计数法计算主轴和塔架的疲劳损伤
tic;  % 记录开始时间
fprintf('进行雨流计数法计算...\n');

% 对所有风机进行雨流计数，并记录循环发生的时间索引
[r_shaft_cycles, r_shaft_amplitudes, r_shaft_means, r_shaft_times] = rainflowForAllTurbines(shaft_load, total_time, num_turbines);
[r_tower_cycles, r_tower_amplitudes, r_tower_means, r_tower_times] = rainflowForAllTurbines(tower_load, total_time, num_turbines);

%% 应用Goodman曲线修正
fprintf('应用Goodman曲线修正...\n');

L_shaft = applyGoodmanCorrection(r_shaft_amplitudes, r_shaft_means, sigma_b, num_turbines);
L_tower = applyGoodmanCorrection(r_tower_amplitudes, r_tower_means, sigma_b, num_turbines);

%% 计算每个时间步长的累计疲劳损伤值
fprintf('计算累计疲劳损伤值...\n');

Damage_shaft_time = calculateCumulativeDamage(L_shaft, r_shaft_cycles, r_shaft_times, C_shaft, m_shaft, total_time, num_turbines);
Damage_tower_time = calculateCumulativeDamage(L_tower, r_tower_cycles, r_tower_times, C_tower, m_tower, total_time, num_turbines);

%% 计算每台风机的等效疲劳载荷
fprintf('计算等效疲劳载荷...\n');

N_design = 42565440.4361;  % 风电风机的设计寿命
LN_shaft = calculateEquivalentLoad(L_shaft, r_shaft_cycles, m_shaft, N_design, num_turbines);
LN_tower = calculateEquivalentLoad(L_tower, r_tower_cycles, m_tower, N_design, num_turbines);

toc;  % 记录结束时间

%% 可视化选定风机的累计疲劳损伤随时间变化
selected_turbines = [5, 10, 50, 75, 100];  % 选择几个风机进行可视化
plotCumulativeDamage(time, Damage_shaft_time, Damage_tower_time, selected_turbines);

%% 比较计算结果与理论输出
compareResults(Damage_shaft_time(end, :), theoretical_shaft_Damage, selected_turbines, '主轴累计疲劳损伤值');
compareResults(Damage_tower_time(end, :), theoretical_tower_Damage, selected_turbines, '塔架累计疲劳损伤值');
compareResults(LN_shaft, theoretical_shaft_LN, selected_turbines, '主轴等效疲劳载荷');
compareResults(LN_tower, theoretical_tower_LN, selected_turbines, '塔架等效疲劳载荷');

%% 辅助函数定义

% 读取疲劳评估数据函数
function [time, load, theoretical_LN, theoretical_Damage] = readLoadData(filename, sheetName, total_time, num_turbines)
    full_data = readmatrix(filename, 'Sheet', sheetName);
    time = full_data(1:total_time, 1);  % 提取时间列
    load = full_data(1:total_time, 2:num_turbines+1);  % 加载数据 (100秒 × 100风机)
    theoretical_LN = full_data(total_time+1, 2:num_turbines+1);      % 理论等效疲劳载荷
    theoretical_Damage = full_data(total_time+2, 2:num_turbines+1);  % 理论累计疲劳损伤值
end

% 雨流计数法计算，记录循环发生的时间索引
function [rainflow_cycles, rainflow_amplitudes, rainflow_means, rainflow_times] = rainflowForAllTurbines(load, total_time, num_turbines)
    rainflow_cycles = cell(1, num_turbines);
    rainflow_amplitudes = cell(1, num_turbines);
    rainflow_means = cell(1, num_turbines);
    rainflow_times = cell(1, num_turbines);
    
    for turbine = 1:num_turbines
        [cycles, amplitudes, means, times] = rainflowCounting(load(:, turbine), total_time);
        rainflow_cycles{turbine} = cycles;
        rainflow_amplitudes{turbine} = amplitudes;
        rainflow_means{turbine} = means;
        rainflow_times{turbine} = times;
    end
end

% 应用 Goodman 曲线修正
function L = applyGoodmanCorrection(amplitudes, means, sigma_b, num_turbines)
    L = cell(1, num_turbines);
    for turbine = 1:num_turbines
        Sai = amplitudes{turbine};  % 载荷幅值
        Smi = means{turbine};       % 载荷均值
        L{turbine} = Sai ./ (1 - Smi / sigma_b);  % Goodman 修正公式: Li = Sai / (1 - Smi / sigma_b)
    end
end

% 计算累计疲劳损伤值
function Damage_time = calculateCumulativeDamage(L, cycles, times, C, m, total_time, num_turbines)
    Damage_time = zeros(total_time, num_turbines);
    for turbine = 1:num_turbines
        Li = L{turbine};
        ni = cycles{turbine};
        ti = times{turbine};  % 循环发生的时间索引
        Nfi = C ./ (Li.^m);   % 对应载荷下的疲劳寿命
        Di = ni ./ Nfi;       % 每个循环的损伤增量

        % 初始化每个时间步的损伤值
        cumulative_damage = zeros(total_time, 1);
        for i = 1:length(Di)
            t_index = ti(i);  % 获取循环发生的时间索引
            cumulative_damage(t_index:end) = cumulative_damage(t_index:end) + Di(i);  % 损伤累加
        end
        Damage_time(:, turbine) = cumulative_damage;
    end
end

% 计算等效疲劳载荷
function LN = calculateEquivalentLoad(L, cycles, m, N_design, num_turbines)
    LN = zeros(1, num_turbines);
    for turbine = 1:num_turbines
        Li = L{turbine};         % Goodman 修正后的载荷幅值
        ni = cycles{turbine};    % 对应的循环次数
        LN(turbine) = (sum((Li.^m) .* ni) / N_design)^(1/m);
    end
end


% 绘制选定风机的累计疲劳损伤随时间变化
function plotCumulativeDamage(time, Damage_shaft_time, Damage_tower_time, selected_turbines)
    figure('Name', '选定风机的累计疲劳损伤随时间变化', 'NumberTitle', 'off');
    for i = 1:length(selected_turbines)
        turbine = selected_turbines(i);
        
        subplot(length(selected_turbines), 2, 2*i-1);
        plot(time, Damage_shaft_time(:, turbine), '-b', 'LineWidth', 1.5);
        title(['风机 ', num2str(turbine), ' 主轴累计疲劳损伤']);
        xlabel('时间 (秒)');
        ylabel('损伤值');
        grid on;
        
        subplot(length(selected_turbines), 2, 2*i);
        plot(time, Damage_tower_time(:, turbine), '-r', 'LineWidth', 1.5);
        title(['风机 ', num2str(turbine), ' 塔架累计疲劳损伤']);
        xlabel('时间 (秒)');
        ylabel('损伤值');
        grid on;
    end
end

% 比较计算结果与理论输出
function compareResults(computed_values, theoretical_values, selected_turbines, comparison_label)
    disp([comparison_label, ' 对比（计算值 vs 理论值）:']);
    comparison_table = table(computed_values(selected_turbines)', ...
        theoretical_values(selected_turbines)', ...
        'VariableNames', {'计算值', '理论值'}, ...
        'RowNames', arrayfun(@num2str, selected_turbines, 'UniformOutput', false));
    disp(comparison_table);
end

% 雨流计数法函数，记录循环发生的时间索引
function [cycles, amplitudes, means, times] = rainflowCounting(signal, total_time)
    signal = signal(:)';  % 确保信号为行向量
    signal = signal([true, diff(signal) ~= 0]);  % 去除连续重复点

    extrema = getExtrema(signal);  % 获取极值点

    % 获取极值点对应的时间索引
    time_indices = getTimeIndices(signal, extrema, total_time);

    % 从最高波峰或最低波谷处开始
    [~, maxPeakIdx] = max(extrema);  % 找到最高波峰的索引
    [~, minValleyIdx] = min(extrema);  % 找到最低波谷的索引

    if maxPeakIdx <= minValleyIdx
        startIdx = maxPeakIdx;
    else
        startIdx = minValleyIdx;
    end

    % 重新组织序列
    extrema = [extrema(startIdx:end), extrema(1:startIdx-1)];
    time_indices = [time_indices(startIdx:end), time_indices(1:startIdx-1)];

    % 初始化变量
    cycles = [];       % 存储循环次数
    amplitudes = [];   % 存储循环幅值
    means = [];        % 存储循环均值
    times = [];        % 存储循环发生的时间索引

    % 循环识别和移除
    while length(extrema) >= 3
        idx = 1;  % 起始索引
        while idx <= length(extrema) - 2
            S1 = extrema(idx);
            S2 = extrema(idx + 1);
            S3 = extrema(idx + 2);

            delta_S1 = abs(S1 - S2);
            delta_S2 = abs(S2 - S3);

            if delta_S1 <= delta_S2
                % 识别出一个有效循环
                Sai = delta_S1;  % 载荷幅值
                Smi = (S1 + S2) / 2;  % 载荷均值

                % 循环发生的时间索引，取参与循环点的较大时间索引
                cycle_time = max(time_indices(idx), time_indices(idx + 1));

                % 记录循环数据
                cycles(end + 1) = 1;  % 完整循环计为1
                amplitudes(end + 1) = Sai;
                means(end + 1) = Smi;
                times(end + 1) = cycle_time;  % 记录循环发生的时间索引

                % 移除已识别的循环点（S1 和 S2）
                extrema(idx:idx+1) = [];
                time_indices(idx:idx+1) = [];

                % 在移除后，重新从序列起始位置开始
                idx = 1;
            else
                % 未识别出循环，索引加1，继续向前
                idx = idx + 1;
            end
        end
        % 当无法再识别循环时，退出循环
        break;
    end
end

% 获取信号中的峰值和谷值
function extrema = getExtrema(signal)
    extrema = [];

    for i = 2:length(signal) - 1
        % 找到波峰或波谷，并记录下来
        if (signal(i) > signal(i-1) && signal(i) > signal(i+1)) || ...
           (signal(i) < signal(i-1) && signal(i) < signal(i+1))
            extrema(end + 1) = signal(i);  % 识别波峰或波谷
        end
    end

    % 添加起点和终点作为极值点
    extrema = [signal(1), extrema, signal(end)];
end

% 获取极值点对应的时间索引
function time_indices = getTimeIndices(signal, extrema, total_time)
    time_indices = zeros(size(extrema));
    for i = 1:length(extrema)
        % 在原信号中找到与当前极值点相等的值，获取其对应的时间索引
        idx = find(signal == extrema(i), 1, 'first');
        time_indices(i) = min(idx, total_time);  % 确保索引不超过总时间
    end
end