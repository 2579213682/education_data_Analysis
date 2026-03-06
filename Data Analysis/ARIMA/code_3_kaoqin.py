import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体和可视化样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. 加载数据
print("正在加载考勤数据...")

# 指定文件路径
file_path = r'D:\桌面\数智教育\education_data\3_kaoqin.csv'

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

# 重命名列名使其更易读
column_mapping = {
    'kaoqing_id': '考勤ID',
    'qj_term': '学期',
    'DataDateTime': '考勤时间',
    'ControllerID': '考勤设备ID',
    'controler_name': '考勤类型',
    'control_task_order_id': '考勤任务ID',
    'bf_studentID': '学号',
    'bf_Name': '姓名',
    'cla_Name': '班级名称',
    'bf_classid': '班级ID'
}

# 只重命名存在的列
existing_columns = {}
for old_col, new_col in column_mapping.items():
    if old_col in data.columns:
        existing_columns[old_col] = new_col
    # 也检查列名是否有空格或其他变体
    elif old_col.lower() in [col.lower() for col in data.columns]:
        actual_col = [col for col in data.columns if col.lower() == old_col.lower()][0]
        existing_columns[actual_col] = new_col

data = data.rename(columns=existing_columns)

# 处理缺失值
print("缺失值统计:")
missing_stats = data.isnull().sum()
missing_percent = (data.isnull().sum() / len(data) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数量': missing_stats,
    '缺失比例(%)': missing_percent
})
print(missing_df[missing_df['缺失数量'] > 0])


# 处理考勤时间
def parse_datetime(dt_str):
    """解析日期时间字符串"""
    if pd.isnull(dt_str):
        return pd.NaT

    dt_str = str(dt_str).strip()

    # 尝试多种日期格式
    formats = [
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y.%m.%d %H:%M:%S',
        '%Y/%m/%d %H:%M',
        '%Y-%m-%d %H:%M',
        '%Y%m%d %H:%M:%S',
        '%Y%m%d %H:%M'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except:
            continue

    # 如果都不行，尝试只提取日期
    try:
        return datetime.strptime(dt_str[:10], '%Y/%m/%d')
    except:
        return pd.NaT


# 应用日期解析
print("正在解析考勤时间...")
data['考勤时间'] = data['考勤时间'].apply(parse_datetime)

# 提取日期和时间成分
data['考勤日期'] = data['考勤时间'].dt.date
data['考勤年份'] = data['考勤时间'].dt.year
data['考勤月份'] = data['考勤时间'].dt.month
data['考勤日'] = data['考勤时间'].dt.day
data['考勤星期'] = data['考勤时间'].dt.dayofweek
data['考勤小时'] = data['考勤时间'].dt.hour
data['考勤分钟'] = data['考勤时间'].dt.minute

# 将星期转换为中文
weekday_map = {
    0: '星期一',
    1: '星期二',
    2: '星期三',
    3: '星期四',
    4: '星期五',
    5: '星期六',
    6: '星期日'
}
data['考勤星期中文'] = data['考勤星期'].map(weekday_map)

# 3. 基本信息统计
print("\n=== 基本信息统计 ===")
print(f"1. 总考勤记录数: {len(data):,}")
if '学期' in data.columns:
    print(f"2. 学期范围: {data['学期'].unique()[:5]}")  # 只显示前5个
if '学号' in data.columns:
    print(f"3. 总学生数: {data['学号'].nunique()}")
if '班级名称' in data.columns:
    print(f"4. 总班级数: {data['班级名称'].nunique()}")
if '考勤时间' in data.columns and not data['考勤时间'].isnull().all():
    valid_times = data['考勤时间'].dropna()
    if len(valid_times) > 0:
        print(f"5. 考勤时间范围: {valid_times.min()} 到 {valid_times.max()}")
if '考勤类型' in data.columns:
    print(f"6. 考勤类型: {data['考勤类型'].unique()[:5]}")  # 只显示前5个

# 4. 可视化分析
fig = plt.figure(figsize=(20, 16))

# 4.1 考勤类型分布
plt.subplot(3, 3, 1)
if '考勤类型' in data.columns:
    attendance_type_counts = data['考勤类型'].value_counts()
    if len(attendance_type_counts) > 0:
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166']
        plt.pie(attendance_type_counts.values[:5], labels=attendance_type_counts.index[:5],
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('考勤类型分布(TOP5)')
    else:
        plt.text(0.5, 0.5, '无考勤类型数据', ha='center', va='center')
        plt.title('考勤类型分布')
else:
    plt.text(0.5, 0.5, '无考勤类型列', ha='center', va='center')
    plt.title('考勤类型分布')

# 4.2 各学期考勤记录数量
plt.subplot(3, 3, 2)
if '学期' in data.columns:
    term_counts = data['学期'].value_counts().sort_index()
    if len(term_counts) > 0:
        bars = plt.bar(range(len(term_counts)), term_counts.values, color='lightblue')
        plt.title('各学期考勤记录数量')
        plt.xlabel('学期')
        plt.ylabel('考勤记录数')
        plt.xticks(range(len(term_counts)), term_counts.index, rotation=45, ha='right')
        for bar, count in zip(bars, term_counts.values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha='center', fontsize=9)
    else:
        plt.text(0.5, 0.5, '无学期数据', ha='center', va='center')
        plt.title('学期分布')
else:
    plt.text(0.5, 0.5, '无学期列', ha='center', va='center')
    plt.title('学期分布')

# 4.3 考勤小时分布
plt.subplot(3, 3, 3)
if '考勤小时' in data.columns:
    hour_counts = data['考勤小时'].value_counts().sort_index()
    if len(hour_counts) > 0:
        plt.plot(hour_counts.index, hour_counts.values, 'o-', linewidth=2, markersize=8, color='green')
        plt.fill_between(hour_counts.index, hour_counts.values, alpha=0.3, color='lightgreen')
        plt.title('考勤小时分布')
        plt.xlabel('小时 (0-23)')
        plt.ylabel('考勤记录数')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))

        # 标记高峰期
        if len(hour_counts) > 0:
            peak_hour = hour_counts.idxmax()
            peak_count = hour_counts.max()
            plt.annotate(f'高峰期: {peak_hour}:00\n({peak_count}次)',
                         xy=(peak_hour, peak_count),
                         xytext=(peak_hour + 2, peak_count * 0.8),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=10, color='red')
    else:
        plt.text(0.5, 0.5, '无小时数据', ha='center', va='center')
        plt.title('小时分布')
else:
    plt.text(0.5, 0.5, '无小时列', ha='center', va='center')
    plt.title('小时分布')

# 4.4 考勤星期分布
plt.subplot(3, 3, 4)
if '考勤星期中文' in data.columns:
    weekday_counts = data['考勤星期中文'].value_counts()
    if len(weekday_counts) > 0:
        # 按星期顺序排序
        ordered_days = [day for day in weekday_map.values() if day in weekday_counts.index]
        weekday_counts = weekday_counts.reindex(ordered_days)

        weekday_counts.plot(kind='bar', color='lightcoral')
        plt.title('考勤星期分布')
        plt.xlabel('星期')
        plt.ylabel('考勤记录数')
        plt.xticks(rotation=45)

        for i, count in enumerate(weekday_counts.values):
            plt.text(i, count + 5, str(count), ha='center', fontsize=9)
    else:
        plt.text(0.5, 0.5, '无星期数据', ha='center', va='center')
        plt.title('星期分布')
else:
    plt.text(0.5, 0.5, '无星期列', ha='center', va='center')
    plt.title('星期分布')

# 4.5 班级考勤记录TOP10
plt.subplot(3, 3, 5)
if '班级名称' in data.columns:
    class_counts = data['班级名称'].value_counts().head(10)
    if len(class_counts) > 0:
        plt.barh(class_counts.index, class_counts.values, color='lightblue')
        plt.title('班级考勤记录TOP10')
        plt.xlabel('考勤记录数')
        for i, v in enumerate(class_counts.values):
            plt.text(v + 5, i, str(v), va='center')
    else:
        plt.text(0.5, 0.5, '无班级数据', ha='center', va='center')
        plt.title('班级分布')
else:
    plt.text(0.5, 0.5, '无班级列', ha='center', va='center')
    plt.title('班级分布')

# 4.6 学生考勤记录TOP10
plt.subplot(3, 3, 6)
if '姓名' in data.columns and '学号' in data.columns:
    student_counts = data.groupby(['姓名', '学号']).size().sort_values(ascending=False).head(10)
    if len(student_counts) > 0:
        student_labels = [f"{name}\n({id})" for name, id in student_counts.index]
        plt.barh(student_labels, student_counts.values, color='lightgreen')
        plt.title('学生考勤记录TOP10')
        plt.xlabel('考勤记录数')
        for i, v in enumerate(student_counts.values):
            plt.text(v + 2, i, str(v), va='center')
    else:
        plt.text(0.5, 0.5, '无学生数据', ha='center', va='center')
        plt.title('学生分布')
else:
    plt.text(0.5, 0.5, '无学生列', ha='center', va='center')
    plt.title('学生分布')

# 4.7 考勤月份分布
plt.subplot(3, 3, 7)
if '考勤月份' in data.columns:
    month_counts = data['考勤月份'].value_counts().sort_index()
    if len(month_counts) > 0:
        months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        month_labels = [months[i - 1] for i in month_counts.index if 1 <= i <= 12]
        month_values = [month_counts[i] for i in month_counts.index if 1 <= i <= 12]

        if len(month_labels) > 0:
            plt.bar(month_labels, month_values, color='skyblue')
            plt.title('考勤月份分布')
            plt.xlabel('月份')
            plt.ylabel('考勤记录数')
            plt.xticks(rotation=45)
            for i, v in enumerate(month_values):
                plt.text(i, v + 5, str(v), ha='center', fontsize=9)
        else:
            plt.text(0.5, 0.5, '月份数据异常', ha='center', va='center')
            plt.title('月份分布')
    else:
        plt.text(0.5, 0.5, '无月份数据', ha='center', va='center')
        plt.title('月份分布')
else:
    plt.text(0.5, 0.5, '无月份列', ha='center', va='center')
    plt.title('月份分布')

# 4.8 考勤设备使用情况
plt.subplot(3, 3, 8)
if '考勤设备ID' in data.columns:
    device_counts = data['考勤设备ID'].value_counts().head(8)
    if len(device_counts) > 0:
        plt.bar(range(len(device_counts)), device_counts.values, color='orange')
        plt.title('考勤设备使用TOP8')
        plt.xlabel('设备ID')
        plt.ylabel('使用次数')
        plt.xticks(range(len(device_counts)), device_counts.index, rotation=45)
        for i, v in enumerate(device_counts.values):
            plt.text(i, v + 5, str(v), ha='center', fontsize=9)
    else:
        plt.text(0.5, 0.5, '无设备数据', ha='center', va='center')
        plt.title('设备分布')
else:
    plt.text(0.5, 0.5, '无设备列', ha='center', va='center')
    plt.title('设备分布')

# 4.9 考勤类型时间分布
plt.subplot(3, 3, 9)
if '考勤类型' in data.columns and '考勤小时' in data.columns:
    # 选择前4种考勤类型进行分析
    top_types = data['考勤类型'].value_counts().head(4).index
    hour_type_data = pd.crosstab(data['考勤小时'], data['考勤类型'])

    # 只选择存在的列
    existing_types = [t for t in top_types if t in hour_type_data.columns]
    if len(existing_types) > 0:
        hour_type_data = hour_type_data[existing_types]
        colors_area = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        hour_type_data.plot.area(ax=plt.gca(), alpha=0.7, color=colors_area[:len(existing_types)])
        plt.title('考勤类型时间分布（小时）')
        plt.xlabel('小时')
        plt.ylabel('考勤记录数')
        plt.legend(title='考勤类型')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
    else:
        plt.text(0.5, 0.5, '无类型-小时数据', ha='center', va='center')
        plt.title('类型时间分布')
else:
    plt.text(0.5, 0.5, '缺少类型或小时列', ha='center', va='center')
    plt.title('类型时间分布')

plt.tight_layout()
plt.show()

# 5. 详细统计分析
print("\n=== 详细统计分析 ===")

# 5.1 学期统计
if '学期' in data.columns:
    print("\n1. 学期统计:")
    term_stats = data.groupby('学期').agg({
        '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x),
        '学号': 'nunique' if '学号' in data.columns else lambda x: 0,
        '班级名称': 'nunique' if '班级名称' in data.columns else lambda x: 0
    })
    if '考勤ID' in term_stats.columns:
        term_stats = term_stats.rename(columns={'考勤ID': '总记录数', '学号': '学生数', '班级名称': '班级数'})
    print(term_stats.head(10))

# 5.2 班级统计
if '班级名称' in data.columns:
    print("\n2. 班级考勤统计TOP10:")
    class_stats = data.groupby('班级名称').agg({
        '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x),
        '学号': 'nunique' if '学号' in data.columns else lambda x: 0
    }).sort_values('考勤ID' if '考勤ID' in data.columns else 0, ascending=False).head(10)

    if '考勤ID' in class_stats.columns:
        class_stats.columns = ['总记录数', '学生数']
    print(class_stats)

# 5.3 学生统计
if '姓名' in data.columns and '学号' in data.columns:
    print("\n3. 学生考勤统计TOP10:")
    student_stats = data.groupby(['学号', '姓名']).agg({
        '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x)
    }).sort_values('考勤ID' if '考勤ID' in data.columns else 0, ascending=False).head(10)

    if '考勤ID' in student_stats.columns:
        student_stats.columns = ['总记录数']
    print(student_stats)

# 5.4 考勤类型详细统计
if '考勤类型' in data.columns:
    print("\n4. 考勤类型详细统计:")
    attendance_type_stats = data.groupby('考勤类型').agg({
        '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x),
        '学号': 'nunique' if '学号' in data.columns else lambda x: 0
    })

    if '考勤ID' in attendance_type_stats.columns:
        attendance_type_stats.columns = ['总记录数', '涉及学生数']
        attendance_type_stats['占比(%)'] = (attendance_type_stats['总记录数'] / len(data) * 100).round(2)
    print(attendance_type_stats)

# 5.5 时间分析
if '考勤小时' in data.columns:
    print("\n5. 时间分析:")
    # 按小时统计
    print("按小时统计:")
    for hour in range(6, 20):  # 6点到20点
        hour_data = data[data['考勤小时'] == hour]
        if len(hour_data) > 0:
            print(f"  {hour:02d}:00-{hour:02d}:59: {len(hour_data)} 条记录")
            if '考勤类型' in data.columns:
                # 显示该小时最常见的考勤类型
                top_type = hour_data['考勤类型'].value_counts().head(1)
                for type_name, count in top_type.items():
                    print(f"      最常见的考勤类型: {type_name} ({count}次)")

# 5.6 迟到分析
if '考勤类型' in data.columns:
    late_keywords = ['迟到', '晚到', 'late', '迟']
    late_types = [atype for atype in data['考勤类型'].unique() if
                  any(keyword in str(atype) for keyword in late_keywords)]

    if len(late_types) > 0:
        print("\n6. 迟到分析:")
        late_data = data[data['考勤类型'].isin(late_types)]
        print(f"  总迟到次数: {len(late_data)}")
        print(f"  涉及学生数: {late_data['学号'].nunique() if '学号' in data.columns else 'N/A'}")
        print(f"  涉及班级数: {late_data['班级名称'].nunique() if '班级名称' in data.columns else 'N/A'}")

        # 迟到时间分析
        if '考勤小时' in late_data.columns:
            late_hour_counts = late_data['考勤小时'].value_counts().sort_index()
            print("  迟到时间分布:")
            for hour, count in late_hour_counts.items():
                print(f"    {hour:02d}:00-{hour:02d}:59: {count} 次")

        # 迟到最多的学生
        if '姓名' in data.columns and '学号' in data.columns:
            late_students = late_data.groupby(['学号', '姓名']).size().sort_values(ascending=False).head(5)
            print("  迟到次数最多的学生:")
            for (student_id, name), count in late_students.items():
                print(f"    {name} ({student_id}): {count} 次")

# 6. 创建更多可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 6.1 各学期考勤类型对比
ax1 = axes[0, 0]
if '学期' in data.columns and '考勤类型' in data.columns:
    term_type_counts = pd.crosstab(data['学期'], data['考勤类型'])
    if not term_type_counts.empty:
        # 只取前5种类型，避免图例过多
        top_5_types = data['考勤类型'].value_counts().head(5).index
        term_type_counts = term_type_counts[top_5_types]

        term_type_counts.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('各学期考勤类型对比(TOP5)')
        ax1.set_xlabel('学期')
        ax1.set_ylabel('考勤记录数')
        ax1.legend(title='考勤类型', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, '无学期-类型数据', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('各学期考勤类型对比')
else:
    ax1.text(0.5, 0.5, '缺少学期或类型列', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('各学期考勤类型对比')

# 6.2 考勤热点时段 - 修复这里的问题
ax2 = axes[0, 1]
if '考勤小时' in data.columns and '考勤星期中文' in data.columns:
    # 删除星期列的缺失值
    hour_week_data = data.dropna(subset=['考勤星期中文', '考勤小时'])

    if len(hour_week_data) > 0:
        # 创建交叉表
        hour_day_counts = pd.crosstab(hour_week_data['考勤小时'], hour_week_data['考勤星期中文'])

        # 确保列的顺序是星期一到星期日
        ordered_days = [day for day in weekday_map.values() if day in hour_day_counts.columns]

        if len(ordered_days) > 0:
            # 按顺序重新排列列
            hour_day_counts = hour_day_counts[ordered_days]

            # 确保有数据
            if not hour_day_counts.empty:
                im = ax2.imshow(hour_day_counts.values.T, aspect='auto', cmap='YlOrRd')
                ax2.set_title('考勤热点时段（小时×星期）')
                ax2.set_xlabel('小时 (0-23)')
                ax2.set_ylabel('星期')
                ax2.set_xticks(range(0, 24, 2))
                ax2.set_yticks(range(len(ordered_days)))
                ax2.set_yticklabels(ordered_days)
                plt.colorbar(im, ax=ax2, label='考勤记录数')
            else:
                ax2.text(0.5, 0.5, '无热点时段数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('考勤热点时段')
        else:
            ax2.text(0.5, 0.5, '无完整的星期数据', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('考勤热点时段')
    else:
        ax2.text(0.5, 0.5, '缺少小时或星期数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('考勤热点时段')
else:
    ax2.text(0.5, 0.5, '缺少小时或星期列', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('考勤热点时段')

# 6.3 迟到学生TOP10
ax3 = axes[1, 0]
if '考勤类型' in data.columns and '姓名' in data.columns and '学号' in data.columns:
    late_keywords = ['迟到', '晚到', 'late', '迟']
    late_types = [atype for atype in data['考勤类型'].unique() if
                  any(keyword in str(atype) for keyword in late_keywords)]

    if len(late_types) > 0:
        late_data = data[data['考勤类型'].isin(late_types)]
        if len(late_data) > 0:
            late_students = late_data.groupby(['姓名', '学号']).size().sort_values(ascending=False).head(10)
            if len(late_students) > 0:
                student_labels = [f"{name}\n({id})" for name, id in late_students.index]
                bars = ax3.barh(student_labels, late_students.values, color='lightcoral')
                ax3.set_title('迟到次数TOP10学生')
                ax3.set_xlabel('迟到次数')
                for i, v in enumerate(late_students.values):
                    ax3.text(v + 0.1, i, str(v), va='center')
            else:
                ax3.text(0.5, 0.5, '无迟到学生数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('迟到次数TOP10学生')
        else:
            ax3.text(0.5, 0.5, '无迟到记录', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('迟到次数TOP10学生')
    else:
        ax3.text(0.5, 0.5, '无迟到类型', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('迟到次数TOP10学生')
else:
    ax3.text(0.5, 0.5, '缺少类型或学生列', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('迟到次数TOP10学生')

# 6.4 考勤趋势（按日期）
ax4 = axes[1, 1]
if '考勤日期' in data.columns:
    daily_counts = data.groupby('考勤日期').size()

    if len(daily_counts) > 0:
        # 取最近60天的数据
        if len(daily_counts) > 60:
            daily_counts = daily_counts[-60:]

        ax4.plot(daily_counts.index, daily_counts.values, 'o-', linewidth=1.5, markersize=4, color='blue')

        # 添加移动平均线
        if len(daily_counts) > 7:
            rolling_mean = daily_counts.rolling(window=7).mean()
            ax4.plot(daily_counts.index, rolling_mean.values, 'r-', linewidth=2, label='7日移动平均')

        ax4.set_title('每日考勤记录趋势')
        ax4.set_xlabel('日期')
        ax4.set_ylabel('考勤记录数')
        if len(daily_counts) > 7:
            ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, '无日期数据', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('每日考勤记录趋势')
else:
    ax4.text(0.5, 0.5, '无日期列', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('每日考勤记录趋势')

plt.tight_layout()
plt.show()

# 7. 生成分析报告
print("\n" + "=" * 60)
print("考勤数据分析报告".center(60))
print("=" * 60)

print(f"\n📊 数据概览:")
print(f"  总考勤记录数: {len(data):,} 条")
if '学号' in data.columns:
    print(f"  涉及学生数: {data['学号'].nunique()} 人")
if '班级名称' in data.columns:
    print(f"  涉及班级数: {data['班级名称'].nunique()} 个")
if '考勤时间' in data.columns and not data['考勤时间'].isnull().all():
    valid_times = data['考勤时间'].dropna()
    if len(valid_times) > 0:
        print(
            f"  考勤时间范围: {valid_times.min().strftime('%Y-%m-%d %H:%M:%S')} 到 {valid_times.max().strftime('%Y-%m-%d %H:%M:%S')}")
if '学期' in data.columns:
    print(f"  学期数量: {data['学期'].nunique()} 个")

if '考勤类型' in data.columns:
    print(f"\n🎯 考勤类型分析:")
    attendance_type_counts = data['考勤类型'].value_counts().head(5)
    for atype, count in attendance_type_counts.items():
        percentage = (count / len(data) * 100)
        print(f"  {atype}: {count}次 ({percentage:.1f}%)")

if '考勤小时' in data.columns:
    hour_counts = data['考勤小时'].value_counts()
    if len(hour_counts) > 0:
        peak_hour = hour_counts.idxmax()
        peak_count = hour_counts.max()
        print(f"\n⏰ 时间规律:")
        print(f"  考勤高峰期: {peak_hour}:00 ({peak_count}次记录)")

if '考勤星期中文' in data.columns:
    weekday_counts = data['考勤星期中文'].value_counts()
    if len(weekday_counts) > 0:
        busiest_day = weekday_counts.idxmax()
        busiest_count = weekday_counts.max()
        print(f"  最繁忙的星期: {busiest_day} ({busiest_count}次记录)")

if '考勤月份' in data.columns:
    month_counts = data['考勤月份'].value_counts()
    if len(month_counts) > 0:
        busiest_month = month_counts.idxmax()
        busiest_count = month_counts.max()
        months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        month_name = months[busiest_month - 1] if 1 <= busiest_month <= 12 else f"{busiest_month}月"
        print(f"  最繁忙的月份: {month_name} ({busiest_count}次记录)")

# 迟到分析
if '考勤类型' in data.columns:
    late_keywords = ['迟到', '晚到', 'late', '迟']
    late_types = [atype for atype in data['考勤类型'].unique() if
                  any(keyword in str(atype) for keyword in late_keywords)]

    if len(late_types) > 0:
        late_data = data[data['考勤类型'].isin(late_types)]
        print(f"\n🚨 迟到情况:")
        print(f"  总迟到次数: {len(late_data)} 次")
        print(f"  迟到率: {len(late_data) / len(data) * 100:.2f}%")

        if '姓名' in data.columns and '学号' in data.columns and len(late_data) > 0:
            late_students = late_data.groupby(['学号', '姓名']).size().sort_values(ascending=False)
            if len(late_students) > 0:
                top_late_student = late_students.index[0][1]
                top_late_count = late_students.values[0]
                print(f"  迟到最多的学生: {top_late_student} ({top_late_count}次)")

if '班级名称' in data.columns:
    class_counts = data['班级名称'].value_counts()
    if len(class_counts) > 0:
        top_class = class_counts.index[0]
        top_class_count = class_counts.iloc[0]
        print(f"\n🏫 班级考勤情况:")
        print(f"  考勤最活跃的班级: {top_class} ({top_class_count}次记录)")

if '姓名' in data.columns and '学号' in data.columns:
    student_counts = data.groupby(['姓名', '学号']).size().sort_values(ascending=False)
    if len(student_counts) > 0:
        top_student_name = student_counts.index[0][1]
        top_student_count = student_counts.values[0]
        print(f"\n👤 学生考勤情况:")
        print(f"  考勤最频繁的学生: {top_student_name} ({top_student_count}次记录)")

if '学期' in data.columns:
    print(f"\n📅 学期对比:")
    term_counts = data['学期'].value_counts().head(5)
    for term, count in term_counts.items():
        print(f"  {term}: {count}次记录")

print("\n" + "=" * 60)
print("分析完成！".center(60))
print("=" * 60)

# 8. 保存分析结果
print("\n正在保存分析结果...")
output_file = '考勤数据_分析结果.csv'
try:
    data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"分析结果已保存到: {output_file}")
except Exception as e:
    print(f"保存CSV文件时出错: {e}")

# 9. 保存统计报表
print("正在生成统计报表...")
try:
    with pd.ExcelWriter('考勤数据_详细报表.xlsx', engine='openpyxl') as writer:
        # 原始数据
        data.to_excel(writer, sheet_name='原始数据', index=False)

        # 学期统计
        if '学期' in data.columns:
            term_stats = data.groupby('学期').agg({
                '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x),
                '学号': 'nunique' if '学号' in data.columns else lambda x: 0
            })
            if '考勤ID' in term_stats.columns:
                term_stats = term_stats.rename(columns={'考勤ID': '总记录数', '学号': '学生数'})
            term_stats.to_excel(writer, sheet_name='学期统计')

        # 班级统计
        if '班级名称' in data.columns:
            class_stats_all = data.groupby('班级名称').agg({
                '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x),
                '学号': 'nunique' if '学号' in data.columns else lambda x: 0
            }).sort_values('考勤ID' if '考勤ID' in data.columns else 0, ascending=False)
            if '考勤ID' in class_stats_all.columns:
                class_stats_all.columns = ['总记录数', '学生数']
            class_stats_all.to_excel(writer, sheet_name='班级统计')

        # 学生统计
        if '姓名' in data.columns and '学号' in data.columns:
            student_stats_all = data.groupby(['学号', '姓名']).agg({
                '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x)
            }).sort_values('考勤ID' if '考勤ID' in data.columns else 0, ascending=False)
            if '考勤ID' in student_stats_all.columns:
                student_stats_all.columns = ['总记录数']
            student_stats_all.head(50).to_excel(writer, sheet_name='学生统计')

        # 考勤类型统计
        if '考勤类型' in data.columns:
            attendance_type_stats = data.groupby('考勤类型').agg({
                '考勤ID': 'count' if '考勤ID' in data.columns else lambda x: len(x),
                '学号': 'nunique' if '学号' in data.columns else lambda x: 0
            })
            if '考勤ID' in attendance_type_stats.columns:
                attendance_type_stats.columns = ['总记录数', '涉及学生数']
                attendance_type_stats['占比(%)'] = (attendance_type_stats['总记录数'] / len(data) * 100).round(2)
            attendance_type_stats.to_excel(writer, sheet_name='考勤类型统计')

        # 时间统计
        if '考勤小时' in data.columns:
            hour_stats = data['考勤小时'].value_counts().sort_index().reset_index()
            hour_stats.columns = ['小时', '记录数']
            hour_stats.to_excel(writer, sheet_name='小时统计', index=False)

    print("详细报表已保存到: 考勤数据_详细报表.xlsx")
except Exception as e:
    print(f"保存Excel报表时出错: {e}")

print("\n✅ 考勤数据分析完成！")
print(f"📁 原始数据形状: {data.shape}")