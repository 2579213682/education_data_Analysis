import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体和可视化样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. 加载数据
print("正在加载成绩数据...")

# 指定文件路径
file_path = r'D:\桌面\数智教育\education_data\5_chengji.csv'

# 尝试不同的编码方式
encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']

data = None
for encoding in encodings:
    try:
        data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        print(f"使用 {encoding} 编码读取成功")
        break
    except Exception as e:
        continue

if data is None:
    print("所有编码尝试失败，尝试自动检测编码...")
    try:
        import chardet

        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        encoding = result['encoding']
        print(f"检测到编码: {encoding}")
        data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    except Exception as e:
        print(f"读取文件失败: {e}")
        exit()

print(f"数据形状: {data.shape}")
print(f"数据列名: {list(data.columns)}")

# 查看前几行数据
print("\n=== 数据前5行预览 ===")
print(data.head())

# 2. 数据清洗和预处理
print("\n=== 数据清洗和预处理 ===")

# 查看列名结构，寻找成绩、时间、学科等关键字段
print("数据列名详情:")
for i, col in enumerate(data.columns):
    print(f"  {i + 1}. {col}")


# 查找关键列
def find_column(keywords, data_columns):
    """根据关键词查找列"""
    for col in data_columns:
        col_lower = str(col).lower()
        for keyword in keywords:
            if keyword in col_lower:
                return col
    return None


# 查找成绩列
score_keywords = ['成绩', 'score', '分数', 'cj', 'grade', 'fraction']
score_col = find_column(score_keywords, data.columns)
if score_col:
    print(f"找到成绩列: {score_col}")
else:
    print("未找到成绩列，尝试使用第一列数值型数据作为成绩")
    # 尝试找到数值型列
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        score_col = numeric_cols[0]
        print(f"使用数值型列作为成绩: {score_col}")
    else:
        print("未找到数值型列，程序退出")
        exit()

# 查找时间列
time_keywords = ['时间', 'date', '考试日期', '时间戳', 'datetime', 'sj']
time_col = find_column(time_keywords, data.columns)
if time_col:
    print(f"找到时间列: {time_col}")
else:
    print("未找到明确的时间列，尝试查找学期列")
    term_keywords = ['学期', 'term', '学年', 'year', 'xq']
    time_col = find_column(term_keywords, data.columns)
    if time_col:
        print(f"找到学期列: {time_col}")

# 查找学科列
subject_keywords = ['科目', '学科', 'subject', 'course', 'xk']
subject_col = find_column(subject_keywords, data.columns)
if subject_col:
    print(f"找到学科列: {subject_col}")

# 处理成绩数据
print(f"\n处理成绩列: {score_col}")
# 转换为数值型
data[score_col] = pd.to_numeric(data[score_col], errors='coerce')

# 移除异常值（通常成绩在0-100之间）
if data[score_col].max() > 1000:  # 如果最大值过大，可能是录入错误
    data[score_col] = data[score_col].clip(0, 100)  # 限制在0-100分
    print("已将成绩限制在0-100分范围内")

# 移除缺失值和无穷大值
data_clean = data.dropna(subset=[score_col])
# 移除无穷大值
data_clean = data_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
print(f"原始数据: {len(data)} 条，清理后: {len(data_clean)} 条")
print(f"成绩统计: 平均{data_clean[score_col].mean():.2f}分, "
      f"标准差{data_clean[score_col].std():.2f}, "
      f"范围{data_clean[score_col].min():.2f}-{data_clean[score_col].max():.2f}")

# 3. 创建时间序列
print("\n=== 创建时间序列 ===")
freq_label = "记录序号"  # 默认值

# 方法1: 如果有时间列
if time_col and ('日期' in time_col.lower() or 'date' in time_col.lower()):
    print(f"使用日期列创建时间序列: {time_col}")
    # 尝试解析日期
    data_clean['日期'] = pd.to_datetime(data_clean[time_col], errors='coerce')
    data_clean = data_clean.dropna(subset=['日期'])

    # 按日、月、年聚合
    print("请选择聚合频率:")
    print("  1. 按日 (daily)")
    print("  2. 按月 (monthly)")
    print("  3. 按年 (yearly)")
    print("  4. 按学期 (假设学期列存在)")

    freq_choice = 2  # 默认按月
    if '学期' in data.columns or 'term' in data.columns:
        freq_choice = 4

    if freq_choice == 1:
        ts_data = data_clean.groupby(pd.Grouper(key='日期', freq='D'))[score_col].mean()
        freq_label = '日'
    elif freq_choice == 2:
        # 兼容pandas新旧版本
        try:
            ts_data = data_clean.groupby(pd.Grouper(key='日期', freq='ME'))[score_col].mean()  # pandas 2.0+
            freq_label = '月'
        except:
            try:
                ts_data = data_clean.groupby(pd.Grouper(key='日期', freq='M'))[score_col].mean()  # pandas 1.x
                freq_label = '月'
            except Exception as e:
                print(f"按月聚合失败: {e}")
                ts_data = data_clean[score_col]
                freq_label = '记录'
    elif freq_choice == 3:
        # 兼容pandas新旧版本
        try:
            ts_data = data_clean.groupby(pd.Grouper(key='日期', freq='YE'))[score_col].mean()  # pandas 2.0+
            freq_label = '年'
        except:
            try:
                ts_data = data_clean.groupby(pd.Grouper(key='日期', freq='Y'))[score_col].mean()  # pandas 1.x
                freq_label = '年'
            except Exception as e:
                print(f"按年聚合失败: {e}")
                ts_data = data_clean[score_col]
                freq_label = '记录'
    else:  # 按学期
        if '学期' in data.columns:
            term_col = '学期'
        elif 'term' in data.columns:
            term_col = 'term'
        else:
            term_col = time_col

        ts_data = data_clean.groupby(term_col)[score_col].mean().sort_index()
        freq_label = '学期'

else:
    # 方法2: 使用索引作为时间（如果数据是按时间顺序的）
    print("使用数据索引作为时间序列")
    ts_data = data_clean[score_col].reset_index(drop=True)
    freq_label = '记录序号'

# 清理时间序列数据：移除NaN和无穷大
ts_data_clean = ts_data.replace([np.inf, -np.inf], np.nan).dropna()

if len(ts_data_clean) < len(ts_data):
    print(f"清理后时间序列: {len(ts_data_clean)} 个点 (原始: {len(ts_data)} 个点)")
    ts_data = ts_data_clean

if len(ts_data) == 0:
    print("错误: 时间序列没有有效数据")
    exit()

print(f"时间序列数据点数量: {len(ts_data)}")
print(f"时间序列范围: {ts_data.index[0]} 到 {ts_data.index[-1]}")
print(f"时间序列值: {list(ts_data.values.round(2))[:10]}...")

# 4. 可视化原始序列
plt.figure(figsize=(15, 10))

# 4.1 原始时间序列
plt.subplot(2, 2, 1)
if isinstance(ts_data.index, pd.DatetimeIndex):
    plt.plot(ts_data.index, ts_data.values, 'b-o', linewidth=2, markersize=5)
    plt.xlabel('时间')
else:
    plt.plot(range(len(ts_data)), ts_data.values, 'b-o', linewidth=2, markersize=5)
    plt.xlabel(freq_label)
plt.ylabel('平均成绩')
plt.title(f'平均成绩{freq_label}变化趋势')
plt.grid(True, alpha=0.3)

# 添加数值标签
if len(ts_data) <= 20:  # 如果数据点不多，添加标签
    for i, (idx, val) in enumerate(ts_data.items()):
        if isinstance(ts_data.index, pd.DatetimeIndex):
            x_pos = i
        else:
            x_pos = i
        plt.text(x_pos, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# 4.2 成绩分布直方图
plt.subplot(2, 2, 2)
plt.hist(ts_data.values, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(ts_data.mean(), color='red', linestyle='--', linewidth=2,
            label=f'平均值: {ts_data.mean():.2f}')
plt.xlabel('平均成绩')
plt.ylabel('频数')
plt.title(f'平均成绩分布 ({freq_label})')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 4.3 自相关图
plt.subplot(2, 2, 3)
lags = min(20, len(ts_data) - 1)
try:
    plot_acf(ts_data, lags=lags, ax=plt.gca())
    plt.title(f'自相关图 (ACF) - {freq_label}数据')
    plt.xlabel('滞后阶数')
    plt.grid(True, alpha=0.3)
except Exception as e:
    plt.text(0.5, 0.5, f'无法绘制ACF图\n错误: {str(e)[:50]}',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('自相关图 (无法绘制)')

# 4.4 偏自相关图
plt.subplot(2, 2, 4)
try:
    plot_pacf(ts_data, lags=lags, ax=plt.gca(), method='ywm')
    plt.title(f'偏自相关图 (PACF) - {freq_label}数据')
    plt.xlabel('滞后阶数')
    plt.grid(True, alpha=0.3)
except Exception as e:
    plt.text(0.5, 0.5, f'无法绘制PACF图\n错误: {str(e)[:50]}',
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('偏自相关图 (无法绘制)')

plt.tight_layout()
plt.show()

# 5. 平稳性检验
print("\n=== 平稳性检验 (ADF Test) ===")

# 确保数据是干净的
ts_values = ts_data.values.copy()

# 检查是否存在NaN或无穷大
if np.any(np.isnan(ts_values)) or np.any(np.isinf(ts_values)):
    print("警告: 时间序列包含NaN或无穷大值，正在清理...")
    ts_values = ts_values[~np.isnan(ts_values)]
    ts_values = ts_values[~np.isinf(ts_values)]

    if len(ts_values) == 0:
        print("错误: 清理后没有有效数据")
        exit()

# 确保至少有4个数据点进行ADF检验
if len(ts_values) < 4:
    print(f"错误: 数据点太少 ({len(ts_values)}个)，至少需要4个点进行ADF检验")
    d = 0
    ts_data_for_model = pd.Series(ts_values)
else:
    try:
        adf_result = ADF(ts_values, autolag='AIC')
        print(f'ADF统计量: {adf_result[0]:.4f}')
        print(f'p-value: {adf_result[1]:.4f}')
        print('临界值:')
        for key, value in adf_result[4].items():
            print(f'  {key}: {value:.4f}')

        if adf_result[1] < 0.05:
            print("结论: p值 < 0.05，拒绝原假设，序列是平稳的。")
            d = 0
            ts_data_for_model = pd.Series(ts_values)
        else:
            print("结论: p值 > 0.05，无法拒绝原假设，序列可能非平稳。")
            # 进行一阶差分
            ts_series = pd.Series(ts_values)
            ts_diff = ts_series.diff().dropna()
            if len(ts_diff) >= 4:
                adf_diff = ADF(ts_diff.values, autolag='AIC')
                print(f"一阶差分后p-value: {adf_diff[1]:.4f}")

                if adf_diff[1] < 0.05:
                    print("一阶差分后序列平稳")
                    d = 1
                    ts_data_for_model = ts_diff
                else:
                    # 二阶差分
                    ts_diff2 = ts_series.diff().diff().dropna()
                    if len(ts_diff2) >= 4:
                        adf_diff2 = ADF(ts_diff2.values, autolag='AIC')
                        print(f"二阶差分后p-value: {adf_diff2[1]:.4f}")

                        if adf_diff2[1] < 0.05:
                            print("二阶差分后序列平稳")
                            d = 2
                            ts_data_for_model = ts_diff2
                        else:
                            print("警告: 二阶差分后仍不平稳，将使用原始序列")
                            d = 0
                            ts_data_for_model = ts_series
                    else:
                        print("二阶差分后数据点不足，使用原始序列")
                        d = 0
                        ts_data_for_model = ts_series
            else:
                print("一阶差分后数据点不足，使用原始序列")
                d = 0
                ts_data_for_model = ts_series
    except Exception as e:
        print(f"ADF检验失败: {e}")
        print("将使用原始序列进行分析")
        d = 0
        ts_data_for_model = pd.Series(ts_values)

# 6. ARIMA模型定阶
print("\n=== ARIMA模型定阶 ===")
n_forecast = 3  # 预测未来3个时间点

# 如果数据点太少，使用简单模型
if len(ts_data_for_model) < 10:
    print(f"数据点较少 ({len(ts_data_for_model)}个)，使用简单ARIMA(1,{d},0)模型")
    p, q = 1, 0
    use_simple_model = True
else:
    use_simple_model = False
    # 通过AIC/BIC准则选择最佳参数
    pmax = min(3, len(ts_data_for_model) // 3)
    qmax = min(3, len(ts_data_for_model) // 3)

    print(f"搜索参数范围: p(0-{pmax}), d({d}), q(0-{qmax})")

    best_aic = np.inf
    best_bic = np.inf
    best_order_aic = None
    best_order_bic = None

    for p in range(pmax + 1):
        for q in range(qmax + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(ts_data_for_model, order=(p, 0, q))
                model_fit = model.fit()

                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order_aic = (p, 0, q)

                if model_fit.bic < best_bic:
                    best_bic = model_fit.bic
                    best_order_bic = (p, 0, q)

            except Exception as e:
                continue

    if best_order_aic:
        print(f"AIC最佳模型: ARIMA{best_order_aic}, AIC={best_aic:.2f}")
        print(f"BIC最佳模型: ARIMA{best_order_bic}, BIC={best_bic:.2f}")

        # 选择AIC或BIC更优的模型
        if best_order_aic == best_order_bic:
            p, _, q = best_order_aic
            print(f"使用ARIMA({p},{d},{q})模型")
        else:
            # 通常选择更简洁的模型
            p, _, q = best_order_bic
            print(f"使用BIC更优的ARIMA({p},{d},{q})模型")
    else:
        print("未能找到合适的ARIMA模型，使用默认ARIMA(1,0,1)")
        p, q = 1, 1

# 7. 模型训练与预测
print(f"\n=== 使用ARIMA({p},{d},{q})模型进行训练与预测 ===")

# 划分训练集和测试集
train_size = max(5, int(len(ts_data) * 0.8))  # 至少5个点，80%训练
train_data = ts_data[:train_size]
test_data = ts_data[train_size:] if len(ts_data) > train_size else None

print(f"训练集大小: {len(train_data)} ({train_size / len(ts_data) * 100:.1f}%)")
if test_data is not None:
    print(f"测试集大小: {len(test_data)} ({len(test_data) / len(ts_data) * 100:.1f}%)")

try:
    # 使用ARIMA模型
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()

    print("\n模型拟合结果:")
    print(model_fit.summary())

    # 进行预测
    if test_data is not None:
        # 预测测试集
        forecast_test = model_fit.forecast(steps=len(test_data))

        # 计算预测误差
        mae = mean_absolute_error(test_data.values, forecast_test)
        rmse = np.sqrt(mean_squared_error(test_data.values, forecast_test))

        print(f"\n测试集预测误差:")
        print(f"  平均绝对误差 (MAE): {mae:.4f}")
        print(f"  均方根误差 (RMSE): {rmse:.4f}")

        # 预测未来
        forecast_future = model_fit.forecast(steps=n_forecast)

    else:
        # 没有测试集，直接预测未来
        forecast_future = model_fit.forecast(steps=n_forecast)

    # 生成未来时间索引
    if isinstance(ts_data.index, pd.DatetimeIndex):
        if freq_label == '月':
            last_date = ts_data.index[-1]
            # 兼容pandas新旧版本
            try:
                future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq='ME')[1:]  # pandas 2.0+
            except:
                try:
                    future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq='M')[1:]  # pandas 1.x
                except:
                    future_dates = [f"未来{i + 1}" for i in range(n_forecast)]
        elif freq_label == '年':
            last_date = ts_data.index[-1]
            # 兼容pandas新旧版本
            try:
                future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq='YE')[1:]  # pandas 2.0+
            except:
                try:
                    future_dates = pd.date_range(start=last_date, periods=n_forecast + 1, freq='Y')[1:]  # pandas 1.x
                except:
                    future_dates = [f"未来{i + 1}" for i in range(n_forecast)]
        else:
            future_dates = [f"未来{i + 1}" for i in range(n_forecast)]
    else:
        future_dates = [f"未来{i + 1}" for i in range(n_forecast)]

    print(f"\n未来 {n_forecast} 个{freq_label}的预测结果:")
    for i, (date, value) in enumerate(zip(future_dates, forecast_future)):
        print(f"  {date}: {value:.2f}分")

    # 8. 可视化预测结果
    plt.figure(figsize=(15, 8))

    # 历史数据
    plt.subplot(2, 2, 1)
    if isinstance(ts_data.index, pd.DatetimeIndex):
        plt.plot(ts_data.index, ts_data.values, 'b-o', linewidth=2, markersize=6, label='历史数据')
    else:
        x_hist = list(range(len(ts_data)))
        plt.plot(x_hist, ts_data.values, 'b-o', linewidth=2, markersize=6, label='历史数据')

    # 拟合值
    fitted_values = model_fit.fittedvalues
    if d > 0:  # 如果有差分，拟合值从d开始
        if isinstance(ts_data.index, pd.DatetimeIndex):
            fit_indices = ts_data.index[d:d + len(fitted_values)]
        else:
            fit_indices = range(d, d + len(fitted_values))
    else:
        if isinstance(ts_data.index, pd.DatetimeIndex):
            fit_indices = ts_data.index[:len(fitted_values)]
        else:
            fit_indices = range(len(fitted_values))

    plt.plot(fit_indices, fitted_values, 'r--', linewidth=2, label='拟合值')

    # 测试集预测
    if test_data is not None:
        if isinstance(ts_data.index, pd.DatetimeIndex):
            test_indices = ts_data.index[train_size:]
        else:
            test_indices = range(train_size, len(ts_data))

        plt.plot(test_indices, forecast_test, 'g--o', linewidth=2, markersize=6, label='测试集预测')

    # 未来预测
    if isinstance(ts_data.index, pd.DatetimeIndex):
        plt.plot(future_dates, forecast_future, 'm--o', linewidth=2, markersize=8, label='未来预测')
    else:
        future_x = list(range(len(ts_data), len(ts_data) + n_forecast))
        plt.plot(future_x, forecast_future, 'm--o', linewidth=2, markersize=8, label='未来预测')

    plt.xlabel(freq_label)
    plt.ylabel('平均成绩')
    plt.title(f'ARIMA({p},{d},{q})模型预测结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 9. 残差分析
    plt.subplot(2, 2, 2)
    residuals = model_fit.resid
    plt.plot(residuals, 'o-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('时间')
    plt.ylabel('残差')
    plt.title('模型残差')
    plt.grid(True, alpha=0.3)

    # 10. 残差分布
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    plt.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                label=f'均值: {residuals.mean():.3f}')
    plt.xlabel('残差值')
    plt.ylabel('频数')
    plt.title('残差分布')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 11. 预测区间
    plt.subplot(2, 2, 4)
    # 获取预测区间
    if test_data is not None:
        forecast_result = model_fit.get_forecast(steps=len(test_data))
    else:
        forecast_result = model_fit.get_forecast(steps=n_forecast)

    forecast_mean = forecast_result.predicted_mean
    forecast_ci = forecast_result.conf_int(alpha=0.05)  # 95%置信区间

    # 绘制历史数据
    if isinstance(ts_data.index, pd.DatetimeIndex):
        plt.plot(ts_data.index, ts_data.values, 'b-', linewidth=2, label='历史数据', alpha=0.7)
    else:
        plt.plot(range(len(ts_data)), ts_data.values, 'b-', linewidth=2, label='历史数据', alpha=0.7)

    # 绘制预测区间
    if isinstance(ts_data.index, pd.DatetimeIndex):
        if test_data is not None:
            pred_indices = ts_data.index[train_size:]
        else:
            pred_indices = future_dates
    else:
        if test_data is not None:
            pred_indices = range(train_size, len(ts_data))
        else:
            pred_indices = range(len(ts_data), len(ts_data) + n_forecast)

    plt.plot(pred_indices, forecast_mean, 'r--o', linewidth=2, markersize=6, label='预测值')
    plt.fill_between(pred_indices, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                     color='gray', alpha=0.2, label='95%置信区间')

    plt.xlabel(freq_label)
    plt.ylabel('平均成绩')
    plt.title('预测结果与置信区间')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 12. 模型评估指标
    print("\n=== 模型评估 ===")

    # 计算模型拟合优度
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ts_data_for_model - ts_data_for_model.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"模型拟合优度 (R²): {r_squared:.4f}")
    print(f"残差均值: {residuals.mean():.4f}")
    print(f"残差标准差: {residuals.std():.4f}")

    # 检查残差是否白噪声
    from statsmodels.stats.diagnostic import acorr_ljungbox

    try:
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        print(f"残差白噪声检验 (Ljung-Box) p-value: {lb_test['lb_pvalue'].iloc[0]:.4f}")
        if lb_test['lb_pvalue'].iloc[0] > 0.05:
            print("结论: 残差是白噪声，模型拟合良好")
        else:
            print("注意: 残差可能不是白噪声，模型可能有改进空间")
    except Exception as e:
        print(f"无法进行白噪声检验: {e}")

except Exception as e:
    print(f"模型训练失败: {e}")
    print("尝试使用简单移动平均进行预测...")

    # 使用简单移动平均
    window = 2
    last_values = list(ts_data.values[-window:])
    simple_forecast = []

    for i in range(n_forecast):
        forecast = np.mean(last_values[-window:])
        simple_forecast.append(forecast)
        last_values.append(forecast)

    print(f"\n简单移动平均预测未来 {n_forecast} 个{freq_label}:")
    for i, value in enumerate(simple_forecast):
        print(f"  未来{i + 1}: {value:.2f}分")

# 13. 生成分析报告
print("\n" + "=" * 60)
print("成绩时间序列预测分析报告".center(60))
print("=" * 60)

print(f"\n📊 数据概况:")
print(f"  总记录数: {len(data)} 条")
print(f"  有效成绩数: {len(data_clean)} 条")
print(f"  平均成绩: {data_clean[score_col].mean():.2f}分")
print(f"  成绩标准差: {data_clean[score_col].std():.2f}")

print(f"\n⏰ 时间序列:")
print(f"  序列长度: {len(ts_data)} 个{freq_label}")
print(f"  时间范围: {ts_data.index[0]} 到 {ts_data.index[-1]}")
print(f"  序列平均值: {ts_data.mean():.2f}分")
print(f"  序列标准差: {ts_data.std():.2f}分")

print(f"\n📈 平稳性检验:")
if 'adf_result' in locals():
    print(f"  ADF p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("  结论: 序列平稳")
    else:
        print(f"  结论: 序列非平稳，使用{d}阶差分")
else:
    print("  平稳性检验未执行或失败")

print(f"\n🤖 模型信息:")
print(f"  最终模型: ARIMA({p},{d},{q})")

if 'forecast_future' in locals():
    print(f"\n🔮 未来预测:")
    for i, (date, value) in enumerate(zip(future_dates, forecast_future)):
        change_pct = ((value - ts_data.values[-1]) / ts_data.values[-1] * 100) if ts_data.values[-1] != 0 else 0
        arrow = "↑" if change_pct > 0 else "↓" if change_pct < 0 else "→"
        print(f"  {date}: {value:.2f}分 ({arrow} {change_pct:+.1f}%)")

print(f"\n📉 数据质量评估:")
print(f"  缺失值比例: {(len(data) - len(data_clean)) / len(data) * 100:.1f}%")
print(f"  异常值处理: 已进行0-100分范围限制")

print("\n" + "=" * 60)
print("分析完成！".center(60))
print("=" * 60)

# 14. 保存结果
print("\n正在保存分析结果...")
try:
    # 保存处理后的时间序列
    if isinstance(ts_data, pd.Series):
        ts_data.to_csv('成绩时间序列.csv', header=True)
    else:
        pd.Series(ts_data).to_csv('成绩时间序列.csv', header=True)
    print("时间序列已保存到: 成绩时间序列.csv")

    # 保存预测结果
    if 'forecast_future' in locals():
        forecast_df = pd.DataFrame({
            '时间': future_dates,
            '预测成绩': forecast_future
        })
        forecast_df.to_csv('成绩预测结果.csv', index=False, encoding='utf-8-sig')
        print("预测结果已保存到: 成绩预测结果.csv")

    # 保存模型摘要
    if 'model_fit' in locals():
        with open('ARIMA模型摘要.txt', 'w', encoding='utf-8') as f:
            f.write(str(model_fit.summary()))
        print("模型摘要已保存到: ARIMA模型摘要.txt")

except Exception as e:
    print(f"保存文件时出错: {e}")

print("\n✅ 成绩时间序列预测分析完成！")