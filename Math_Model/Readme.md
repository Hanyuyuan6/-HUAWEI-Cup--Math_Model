注意：
运行本代码需要一些优化工具，请根据提示安装
请将本文件中所有数据和代码文件都加入路径(避免其他路径有重名文件)
附件3-噪声和延迟作用下的采集数据.mat 加载到matlab后被更名为 data_TS_WF_noise

文件夹中数据文件解释：
文件夹 question_1 是问题一的代码，里面仅有一个文件 question_1.m 是问题一答案模型

文件夹 question_2 是问题二的代码，其中，question_2.m 是本题答案模型；
question_2_tradition.m  是完全基于物理机理的计算，作为我们本题答案的对比模型
for_save.m 是把对100 台风机预测的的应力/扭矩计算结果列入附件 6表格中
Ct_Cp_model_from_Q2.m 是基于问题二的模型的多元线性回归模型得到并保存Ct和Cp

文件夹 question_3 是问题三的代码和数据， 其中，question_3.m 是本题答案模型；
question_3_local_OPT.m 是陷入局部最优解的模型，作为本题答案的对比模型；
用于画图的数据文件：
single_total_damage_avg_total_3.mat 数据是选定5个风机的基于平均功率分配的累积疲劳损伤；
single_total_damage_opt_total_3.mat 数据是选定5个风机的基于优化后的功率分配的累积疲劳损伤；
variance_avg_3.mat 数据是100个风机的基于平均功率分配的方差
variance_opt_3.mat 数据是100个风机的基于优化后的功率分配的方差
weighted_total_damage_avg_3.mat 数据是加权后100个风机的基于平均功率分配的总累积疲劳损伤
weighted_total_damage_opt_3.mat 数据是加权后100个风机的基于优化后的功率分配的总累积疲劳损伤

文件夹 question_4 是问题三的代码和数据，其中，question_4.m 是本题答案模型；
Calcul_state_transfer.m 是基于附件一去计算合理的状态转移矩阵；
PCA_for_weight.m 是基于无噪声的附件二数据去计算累积疲劳损伤中变量的权重；
compare_in_different_condition.m 是展示所建模型对噪声和延迟的抑制能力

用于画图的数据文件：
single_total_damage_avg_total_4.mat 数据是选定5个风机的基于平均功率分配的累积疲劳损伤；
single_total_damage_opt_total_4.mat 数据是选定5个风机的基于优化后的功率分配的累积疲劳损伤；
variance_avg_4.mat 数据是100个风机的基于平均功率分配的方差
variance_opt_4.mat 数据是100个风机的基于优化后的功率分配的方差
weighted_total_damage_avg_4.mat 数据是加权后100个风机的基于平均功率分配的总累积疲劳损伤
weighted_total_damage_opt_4.mat 数据是加权后100个风机的基于优化后的功率分配的总累积疲劳损伤

其他数据文件解释：
结果动图展示：
question_3_基于粒子群优化.avi
question_3_陷入局部最优(平均).avi
question_4_优化功率分配动态展示.avi

mdl_Ct.mat；mdl_Cp.mat；feature_mean_std.mat---------for_save.m 生成的数据
Torque_Thrust_Matrices.mat---------附件6数据
transition_matrix_shaft.mat；transition_matrix_tower.mat---------Calcul_state_transfer.m生成的数据
results_baseline.mat 无滤波和预测的模型结果
results_arima.mat 仅ARIMA预测的模型结果
results_kalman.mat 仅卡尔曼滤波的模型结果
results_full.mat 滤波和预测结合的模型结果


题目数据：
附件1-疲劳评估数据.xls；附件2-风电机组采集数据.mat；附件3-噪声和延迟作用下的采集数据.mat；附件4-噪声和延迟作用下的采集数据.xlsx；

问题一，问题二答案：
附件5-问题一答案表.xlsx；附件6-问题二答案表.xlsx


