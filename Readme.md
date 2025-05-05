# **📌 华为杯数学建模竞赛项目说明**

## **🔹 运行说明**
- **运行本代码需要一些优化工具**，请根据提示安装。
- **请将本文件中所有数据和代码文件都加入当前路径**，避免因路径问题导致文件无法访问。
- **MATLAB 数据加载说明**：
  - `附件3-噪声和延迟作用下的采集数据.mat` 加载到 MATLAB 后，变量名称自动更改为 `data_TS_WF_noise`。

---

## **📂 文件夹结构及内容**
项目文件夹包含四个主要部分，分别对应竞赛中的四个问题，每个文件夹中包含了代码、数据及相关说明。

### **📁 Question 1 —— 问题一**
- 📜 `question_1.m`：**问题一**的核心代码。

---

### **📁 Question 2 —— 问题二**
- 📜 `question_2.m`：**问题二**的核心代码。
- 📜 `question_2_tradition.m`：**基于物理机理的计算**，作为对比模型。
- 📜 `for_save.m`：**将对 100 台风机预测的应力/扭矩计算结果列入附件 6 表格**。
- 📜 `Ct_Cp_model_from_Q2.m`：**基于问题二模型的多元线性回归模型**，用于保存 Ct 和 Cp。

---

### **📁 Question 3 —— 问题三**
- 📜 `question_3.m`：**问题三**的核心代码。
- 📜 `question_3_local_OPT.m`：**陷入局部最优解的模型**，作为对比模型。
- 📊 **用于画图的数据文件**：
  - `single_total_damage_avg_total_3.mat`：**5 台风机的累积疲劳损伤（平均功率分配）**。
  - `single_total_damage_opt_total_3.mat`：**5 台风机的累积疲劳损伤（优化功率分配）**。
  - `variance_avg_3.mat`：**100 台风机的方差（平均功率分配）**。
  - `variance_opt_3.mat`：**100 台风机的方差（优化功率分配）**。
  - `weighted_total_damage_avg_3.mat`：**100 台风机的总累积疲劳损伤（平均功率分配）**。
  - `weighted_total_damage_opt_3.mat`：**100 台风机的总累积疲劳损伤（优化功率分配）**。

---

### **📁 Question 4 —— 问题四**
- 📜 `question_4.m`：**问题四**的核心代码。
- 📜 `Calcul_state_transfer.m`：**基于附件 1 计算合理的状态转移矩阵**。
- 📜 `PCA_for_weight.m`：**基于附件 2（无噪声数据）计算累积疲劳损伤变量的权重**。
- 📜 `compare_in_different_condition.m`：**展示所建模型对噪声和延迟的抑制能力**。
- 📊 **用于画图的数据文件**：
  - `single_total_damage_avg_total_4.mat`：**5 台风机的累积疲劳损伤（平均功率分配）**。
  - `single_total_damage_opt_total_4.mat`：**5 台风机的累积疲劳损伤（优化功率分配）**。
  - `variance_avg_4.mat`：**100 台风机的方差（平均功率分配）**。
  - `variance_opt_4.mat`：**100 台风机的方差（优化功率分配）**。
  - `weighted_total_damage_avg_4.mat`：**100 台风机的总累积疲劳损伤（平均功率分配）**。
  - `weighted_total_damage_opt_4.mat`：**100 台风机的总累积疲劳损伤（优化功率分配）**。

---

## **📂 其他数据文件**
### **📂 结果动图**
- 🎞️ `question_3_基于粒子群优化.avi`  
- 🎞️ `question_3_陷入局部最优(平均).avi`  
- 🎞️ `question_4_优化功率分配动态展示.avi`  

### **📂 计算结果**
- 📜 `mdl_Ct.mat`、`mdl_Cp.mat`、`feature_mean_std.mat`：`for_save.m` 生成的数据。
- 📜 `Torque_Thrust_Matrices.mat`：**附件 6 的数据**。
- 📜 `transition_matrix_shaft.mat`、`transition_matrix_tower.mat`：`Calcul_state_transfer.m` 生成的数据。

### **📂 预测与滤波模型结果**
- 📜 `results_baseline.mat`：**无滤波和预测的模型结果**。
- 📜 `results_arima.mat`：**仅 ARIMA 预测的模型结果**。
- 📜 `results_kalman.mat`：**仅卡尔曼滤波的模型结果**。
- 📜 `results_full.mat`：**滤波和预测结合的模型结果**。

---

## **📂 题目数据**
- 📜 `附件1-疲劳评估数据.xls`
- 📜 `附件2-风电机组采集数据.mat`
- 📜 `附件3-噪声和延迟作用下的采集数据.mat`
- 📜 `附件4-噪声和延迟作用下的采集数据.xlsx`

---

## **📂 问题答案**
- 📜 `附件5-问题一答案表.xlsx`
- 📜 `附件6-问题二答案表.xlsx`

---

## **🎯 最后**
- **请确保安装所有必要的优化工具**，并将所有数据和代码文件放入 **同一目录** 避免路径问题。
- **注意**：加载 `.mat` 文件时可能会自动更名，请检查变量名称。

