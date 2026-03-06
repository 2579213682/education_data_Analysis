import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 1. 导入并处理数据
data = pd.read_csv(r'D:\桌面\数智教育\education_data\1_teacher.csv', encoding='utf-8')
print("数据预览（前5行）:")
print(data.head())
print("\n数据列名:", data.columns.tolist())
print(f"总数据条数: {len(data)}")


# 提取学期信息 - 修复学期处理函数
def process_term_for_sorting(term_str):
    """将学期转换为可排序的格式"""
    try:
        parts = term_str.split('-')
        if len(parts) >= 3:
            start_year = int(parts[0])  # 起始年份
            end_year = int(parts[1])  # 结束年份
            term_num = int(parts[2])  # 学期号 (1或2)

            # 第一学期（秋季学期）在8月左右开始，第二学期在2月左右开始
            # 为了正确排序，我们创建一个数值：起始年份*10 + 学期号
            # 这样2014-2015-1变成20141，2014-2015-2变成20142
            return start_year * 10 + term_num
    except:
        return None
    return None


# 应用处理函数
data['sort_key'] = data['term'].apply(process_term_for_sorting)
data = data.dropna(subset=['sort_key'])
data['sort_key'] = data['sort_key'].astype(int)

# 按sort_key排序
data = data.sort_values('sort_key')

# 2. 创建时间序列：统计每个学期的独立教师数量
# 先按term分组，然后按sort_key排序
teacher_count_by_term = data.groupby('term')['bas_id'].nunique()
term_order = sorted(teacher_count_by_term.index, key=lambda x: process_term_for_sorting(x))

# 重新按正确顺序排序
teacher_count_series = pd.Series([teacher_count_by_term[term] for term in term_order],
                                 index=term_order)

print("\n各学期独立教师数量统计（按时间顺序）:")
for i, (term, count) in enumerate(teacher_count_series.items()):
    print(f"  {term}: {count} 人")
print(f"数据点数量: {len(teacher_count_series)}")

# 将序列转换为Pandas Series
ts_data = teacher_count_series.astype(float)
term_labels = list(ts_data.index)

# 3. 数据可视化
plt.figure(figsize=(16, 12))

# 3.1 原始序列
plt.subplot(3, 3, 1)
x_positions = range(len(ts_data))
plt.plot(x_positions, ts_data.values, 'bo-', markersize=8, linewidth=2, label='实际值')
plt.title('各学期独立教师数量变化趋势')
plt.xlabel('学期')
plt.ylabel('教师数量')
plt.xticks(x_positions, term_labels, rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)
for i, (x, y) in enumerate(zip(x_positions, ts_data.values)):
    plt.annotate(f'{int(y)}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9)
plt.legend()

# 4. 基本分析
print("\n=== 时间序列分析 ===")
print(f"数据点数量: {len(ts_data)}")
print(f"学期序列: {term_labels}")
print(f"教师数量: {list(ts_data.values)}")

# 5. 计算基本统计量和增长率
mean_val = ts_data.mean()
std_val = ts_data.std()
min_val = ts_data.min()
max_val = ts_data.max()
print(f"\n基本统计量:")
print(f"平均值: {mean_val:.2f}")
print(f"标准差: {std_val:.2f}")
print(f"最小值: {min_val}")
print(f"最大值: {max_val}")
print(f"极差: {max_val - min_val}")

# 计算学期增长率
growth_rates = []
for i in range(1, len(ts_data)):
    rate = (ts_data.values[i] - ts_data.values[i - 1]) / ts_data.values[i - 1] * 100
    growth_rates.append(rate)
    print(f"从 {term_labels[i - 1]} 到 {term_labels[i]}: {rate:.1f}%")

if growth_rates:
    avg_growth = sum(growth_rates) / len(growth_rates)
    print(f"平均学期增长率: {avg_growth:.1f}%")

# 6. 平稳性检验
print("\n--- 平稳性检验 (ADF Test) ---")
adf_result = ADF(ts_data.values)
print(f'ADF Statistic: {adf_result[0]:.4f}')
print(f'p-value: {adf_result[1]:.4f}')
if adf_result[1] < 0.05:
    print("结论: p值 < 0.05，拒绝原假设，序列是平稳的。")
else:
    print("结论: p值 > 0.05，无法拒绝原假设，序列可能非平稳。")
    # 进行一阶差分
    ts_data_diff = ts_data.diff().dropna()
    adf_result_diff = ADF(ts_data_diff.values)
    print(f"一阶差分后ADF p-value: {adf_result_diff[1]:.4f}")
    if adf_result_diff[1] < 0.05:
        print("结论: 一阶差分后序列平稳。")

# 7. 自相关和偏自相关图 - 修复lags参数
plt.subplot(3, 3, 2)
max_lags = min(10, len(ts_data) - 1)
plot_acf(ts_data, lags=max_lags, ax=plt.gca())
plt.title('自相关图(ACF)')
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(3, 3, 3)
# PACF的lags不能超过样本量的50%
max_lags_pacf = min(4, int(len(ts_data) * 0.5))  # 不超过样本量的50%
plot_pacf(ts_data, lags=max_lags_pacf, ax=plt.gca(), method='ywm')
plt.title('偏自相关图(PACF)')
plt.grid(True, linestyle='--', alpha=0.5)

# 8. ARIMA模型定阶
print("\n=== ARIMA模型定阶与预测 ===")
n_forecast = 3  # 预测未来3个学期

# 使用BIC准则选择最佳p,d,q
pmax = min(3, len(ts_data) // 3)
qmax = min(3, len(ts_data) // 3)
dmax = 2  # 最大差分阶数

print(f"搜索参数范围: p(0-{pmax}), d(0-{dmax}), q(0-{qmax})")

best_bic = np.inf
best_order = None
best_model = None

for p in range(pmax + 1):
    for d in range(dmax + 1):
        for q in range(qmax + 1):
            if p == 0 and d == 0 and q == 0:
                continue
            try:
                model = ARIMA(ts_data, order=(p, d, q))
                model_fit = model.fit()
                bic = model_fit.bic
                if bic < best_bic:
                    best_bic = bic
                    best_order = (p, d, q)
                    best_model = model_fit
            except:
                continue

if best_order is not None:
    print(f"最佳模型阶数 (p,d,q) = {best_order}, BIC = {best_bic:.2f}")

    # 使用最佳模型进行预测
    forecast_result = best_model.get_forecast(steps=n_forecast)
    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)

    # 生成未来学期的标签
    last_term = term_labels[-1]
    last_parts = last_term.split('-')
    last_start_year = int(last_parts[0])
    last_end_year = int(last_parts[1])
    last_term_num = int(last_parts[2])

    future_terms = []
    for i in range(1, n_forecast + 1):
        if last_term_num == 1:
            # 当前是第一学期，下一个是第二学期
            future_terms.append(f"{last_start_year}-{last_end_year}-2")
            last_term_num = 2
        else:
            # 当前是第二学期，下一个是第一学期（新年份）
            future_terms.append(f"{last_start_year + 1}-{last_end_year + 1}-1")
            last_start_year += 1
            last_end_year += 1
            last_term_num = 1

    print(f"\n未来 {n_forecast} 个学期的预测结果:")
    for i, term in enumerate(future_terms):
        mean_val = forecast_mean.iloc[i]
        ci_low = forecast_ci.iloc[i, 0]
        ci_high = forecast_ci.iloc[i, 1]
        print(f"  {term}: {mean_val:.1f} 人 (95% CI: [{ci_low:.1f}, {ci_high:.1f}])")

    # 9. 可视化预测结果
    plt.subplot(3, 3, 4)
    # 历史数据
    x_hist = list(range(len(ts_data)))
    plt.plot(x_hist, ts_data.values, 'bo-', markersize=8, linewidth=2, label='历史数据')

    # 拟合值
    fitted_values = best_model.fittedvalues
    if len(fitted_values) > 0:
        fit_start = best_order[1]  # 差分阶数d
        fit_x = list(range(fit_start, fit_start + len(fitted_values)))
        plt.plot(fit_x, fitted_values, 'r--', linewidth=2, label='拟合值')

    # 预测值
    forecast_x = list(range(len(ts_data), len(ts_data) + n_forecast))
    plt.plot(forecast_x, forecast_mean.values, 'go--', markersize=8, linewidth=2, label='预测值')
    plt.fill_between(forecast_x, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                     color='gray', alpha=0.2, label='95% 置信区间')

    plt.title(f'ARIMA{best_order}模型: 教师数量预测')
    plt.xlabel('学期序号')
    plt.ylabel('教师数量')

    # 设置x轴标签
    all_terms = term_labels + future_terms
    all_x = list(range(len(all_terms)))
    plt.xticks(all_x, all_terms, rotation=45, ha='right')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 10. 残差分析
    plt.subplot(3, 3, 5)
    residuals = best_model.resid
    plt.plot(residuals, 'o-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.title('模型残差')
    plt.xlabel('学期序号')
    plt.ylabel('残差')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 11. 残差直方图
    plt.subplot(3, 3, 6)
    plt.hist(residuals, bins=10, edgecolor='black', alpha=0.7)
    plt.title('残差分布')
    plt.xlabel('残差值')
    plt.ylabel('频数')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')

    # 12. 数据分布和统计
    plt.subplot(3, 3, 7)
    plt.boxplot(ts_data.values, vert=False)
    plt.title('教师数量分布箱线图')
    plt.yticks([])
    plt.xlabel('教师数量')
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')

    # 13. 柱状图
    plt.subplot(3, 3, 8)
    bars = plt.bar(x_hist, ts_data.values, color='skyblue', edgecolor='black')
    plt.title('各学期教师数量柱状图')
    plt.xticks(x_hist, term_labels, rotation=45, ha='right')
    plt.ylabel('教师数量')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')

    # 在柱子上添加数值
    for i, (bar, v) in enumerate(zip(bars, ts_data.values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 str(int(v)), ha='center', fontsize=9)

    # 14. 增长率分析
    plt.subplot(3, 3, 9)
    if len(growth_rates) > 0:
        growth_labels = [f"{term_labels[i]}\n→\n{term_labels[i + 1]}" for i in range(len(growth_rates))]
        colors_growth = ['green' if rate >= 0 else 'red' for rate in growth_rates]

        x_pos = range(len(growth_rates))
        bars = plt.bar(x_pos, growth_rates, color=colors_growth, edgecolor='black')
        plt.title('学期间增长率')
        plt.xlabel('学期区间')
        plt.ylabel('增长率 (%)')
        plt.xticks(x_pos, growth_labels, rotation=45, ha='right', fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5, axis='y')

        # 在柱子上添加数值
        for bar, rate in zip(bars, growth_rates):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            y_text = height + 1 if height >= 0 else height - 5
            plt.text(bar.get_x() + bar.get_width() / 2, y_text,
                     f'{rate:.1f}%', ha='center', va=va, fontsize=8)
    else:
        plt.text(0.5, 0.5, '数据不足\n无法计算增长率',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')

else:
    print("未能找到合适的ARIMA模型，尝试简单预测方法")

    # 简单移动平均预测
    window = 2
    last_values = list(ts_data.values[-window:])
    simple_forecast = []
    for i in range(n_forecast):
        forecast = sum(last_values[-window:]) / window
        simple_forecast.append(forecast)
        last_values.append(forecast)

    # 生成未来学期标签
    last_term = term_labels[-1]
    last_parts = last_term.split('-')
    last_start_year = int(last_parts[0])
    last_end_year = int(last_parts[1])
    last_term_num = int(last_parts[2])

    future_terms = []
    for i in range(1, n_forecast + 1):
        if last_term_num == 1:
            future_terms.append(f"{last_start_year}-{last_end_year}-2")
            last_term_num = 2
        else:
            future_terms.append(f"{last_start_year + 1}-{last_end_year + 1}-1")
            last_start_year += 1
            last_end_year += 1
            last_term_num = 1

    print(f"\n简单移动平均预测未来 {n_forecast} 个学期:")
    for i, term in enumerate(future_terms):
        print(f"  {term}: {simple_forecast[i]:.1f} 人")

plt.tight_layout()
plt.show()

# 15. 输出建议
print("\n=== 分析与建议 ===")
print(f"1. 数据概况: 共{len(ts_data)}个学期数据点 ({term_labels[0]} 到 {term_labels[-1]})")
print(f"2. 教师数量变化范围: {min_val} - {max_val}")
print(f"3. 最近一学期 ({term_labels[-1]}) 教师数量: {ts_data.values[-1]}")

if len(growth_rates) > 0:
    if avg_growth > 0:
        print(f"4. 趋势: 总体呈上升趋势，平均学期增长率 {avg_growth:.1f}%")
    elif avg_growth < 0:
        print(f"4. 趋势: 总体呈下降趋势，平均学期增长率 {avg_growth:.1f}%")
    else:
        print(f"4. 趋势: 基本保持稳定")

print(f"5. 数据点数量: {len(ts_data)}个学期数据")
print(f"6. 教师数量标准差: {std_val:.2f}，数据波动性 {'较大' if std_val / mean_val > 0.2 else '适中'}")

if best_order is not None:
    print(f"7. 推荐ARIMA模型: ARIMA{best_order}")
    print("8. 可以使用该模型进行未来学期的教师数量预测")
    print(f"9. 预测未来学期教师数量:")
    for i, term in enumerate(future_terms):
        mean_val_pred = forecast_mean.iloc[i]
        ci_low = forecast_ci.iloc[i, 0]
        ci_high = forecast_ci.iloc[i, 1]
        print(f"   {term}: {mean_val_pred:.1f} 人 (95%置信区间: {ci_low:.1f} - {ci_high:.1f})")
else:
    print("7. 由于数据特征，ARIMA模型可能不是最佳选择")
    print("8. 可以考虑其他时间序列方法，如指数平滑、Prophet等")

print("\n温馨提示:")
print("1. 时间序列预测结果受历史数据模式影响，实际应用中需结合实际情况综合考虑。")
print("2. 教师数量受多种因素影响，包括学校规模、课程设置、教师流动等。")
print("3. 预测结果仅供参考，建议结合学校发展规划进行调整。")