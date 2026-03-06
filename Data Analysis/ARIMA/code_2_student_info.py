import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体和可视化样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 1. 加载数据
print("正在加载数据...")
data = pd.read_csv(r'D:\桌面\数智教育\education_data\2_student_info.csv', encoding='utf-8-sig')
print(f"数据形状: {data.shape}")
print(f"数据列名: {list(data.columns)}")

# 检查列名
print("\n=== 数据前5行预览 ===")
print(data.head())

# 2. 数据清洗和预处理
print("\n=== 数据清洗和预处理 ===")

# 重命名列名使其更易读
column_mapping = {
    'bf_StudentID': '学号',
    'bf_Name': '姓名',
    'bf_sex': '性别',
    'bf_nation': '民族',
    'bf_BornDate': '出生日期',
    'cla_Name': '班级名称',
    'bf_NativePlace': '籍贯',
    'Bf_ResidenceType': '户口类型',
    'bf_policy': '政治面貌',
    'cla_id': '班级ID',
    'cla_term': '学期',
    'bf_zhusu': '是否住宿',
    'bf_leaveSchool': '是否离校',
    'bf_qinshihao': '寝室号'
}

data = data.rename(columns=column_mapping)

# 处理缺失值
print("缺失值统计:")
print(data.isnull().sum())


# 清理出生日期列
def clean_birth_date(date_val):
    if pd.isnull(date_val):
        return np.nan
    try:
        # 处理各种日期格式
        if isinstance(date_val, (int, float)):
            date_val = str(int(date_val))
        if len(str(date_val)) == 4:  # 只有年份
            return f"{date_val}-01-01"
        elif len(str(date_val)) == 8 and str(date_val).isdigit():  # 20140820格式
            return f"{str(date_val)[:4]}-{str(date_val)[4:6]}-{str(date_val)[6:]}"
        return str(date_val)
    except:
        return np.nan


data['出生日期'] = data['出生日期'].apply(clean_birth_date)
data['出生年份'] = pd.to_datetime(data['出生日期'], errors='coerce').dt.year

# 计算年龄（以2018年为基准，因为数据是2018-2019学期）
data['年龄'] = 2018 - data['出生年份']

# 3. 基本信息统计
print("\n=== 基本信息统计 ===")
print(f"1. 总学生数: {len(data)}")
print(f"2. 学期范围: {data['学期'].unique()}")
print(f"3. 总班级数: {data['班级名称'].nunique()}")

# 4. 可视化分析
fig = plt.figure(figsize=(20, 16))

# 4.1 性别分布
plt.subplot(3, 3, 1)
gender_counts = data['性别'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
        colors=['lightcoral', 'lightblue'], startangle=90)
plt.title('性别分布')

# 4.2 年龄分布
plt.subplot(3, 3, 2)
age_data = data['年龄'].dropna()
plt.hist(age_data, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(age_data.mean(), color='red', linestyle='--', linewidth=2,
            label=f'平均年龄: {age_data.mean():.1f}岁')
plt.xlabel('年龄')
plt.ylabel('人数')
plt.title(f'年龄分布 (最小{age_data.min():.0f}岁, 最大{age_data.max():.0f}岁)')
plt.legend()
plt.grid(True, alpha=0.3)

# 4.3 民族分布
plt.subplot(3, 3, 3)
nation_counts = data['民族'].value_counts().head(10)
plt.barh(nation_counts.index, nation_counts.values, color='lightgreen')
plt.xlabel('人数')
plt.title('民族分布TOP10')
for i, v in enumerate(nation_counts.values):
    plt.text(v + 0.5, i, str(v), va='center')

# 4.4 政治面貌分布
plt.subplot(3, 3, 4)
policy_counts = data['政治面貌'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
plt.pie(policy_counts.values, labels=policy_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=140)
plt.title('政治面貌分布')

# 4.5 户口类型分布
plt.subplot(3, 3, 5)
residence_counts = data['户口类型'].value_counts()
plt.bar(residence_counts.index, residence_counts.values, color=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('户口类型分布')
plt.xlabel('户口类型')
plt.ylabel('人数')
for i, v in enumerate(residence_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center')

# 4.6 住宿情况
plt.subplot(3, 3, 6)
accom_counts = data['是否住宿'].value_counts()
plt.bar(['住宿', '走读'], accom_counts.values, color=['lightblue', 'lightcoral'])
plt.title('住宿情况')
plt.ylabel('人数')
for i, v in enumerate(accom_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center')

# 4.7 班级人数分布
plt.subplot(3, 3, 7)
class_counts = data['班级名称'].value_counts().head(15)
plt.barh(class_counts.index, class_counts.values, color='lightblue')
plt.xlabel('人数')
plt.title('班级人数TOP15')
for i, v in enumerate(class_counts.values):
    plt.text(v + 0.5, i, str(v), va='center')

# 4.8 籍贯分布（TOP15）
plt.subplot(3, 3, 8)
native_counts = data['籍贯'].value_counts().head(15)
plt.barh(native_counts.index, native_counts.values, color='lightgreen')
plt.xlabel('人数')
plt.title('籍贯分布TOP15')
for i, v in enumerate(native_counts.values):
    plt.text(v + 0.5, i, str(v), va='center')

# 4.9 离校情况
plt.subplot(3, 3, 9)
leave_counts = data['是否离校'].value_counts()
plt.bar(['在校', '离校'], leave_counts.values, color=['lightgreen', 'lightcoral'])
plt.title('离校情况')
plt.ylabel('人数')
for i, v in enumerate(leave_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.tight_layout()
plt.show()

# 5. 详细统计分析
print("\n=== 详细统计分析 ===")

# 5.1 按班级统计
print("\n1. 班级统计:")
class_stats = data.groupby('班级名称').agg({
    '学号': 'count',
    '性别': lambda x: (x == '男').sum(),
    '年龄': 'mean'
}).round(2)
class_stats.columns = ['总人数', '男生数', '平均年龄']
print(class_stats.head(10))

# 5.2 按政治面貌统计
print("\n2. 政治面貌统计:")
policy_stats = data.groupby('政治面貌').agg({
    '学号': 'count',
    '性别': lambda x: (x == '男').sum(),
    '年龄': 'mean'
}).round(2)
policy_stats.columns = ['总人数', '男生数', '平均年龄']
print(policy_stats)

# 5.3 按籍贯统计
print("\n3. 籍贯TOP10:")
native_top10 = data['籍贯'].value_counts().head(10)
for place, count in native_top10.items():
    print(f"  {place}: {count}人")

# 5.4 年龄分段统计
print("\n4. 年龄分段统计:")
age_bins = [0, 15, 16, 17, 18, 20, 30]
age_labels = ['<15岁', '15岁', '16岁', '17岁', '18岁', '>18岁']
data['年龄分段'] = pd.cut(data['年龄'], bins=age_bins, labels=age_labels, right=False)
age_group_stats = data['年龄分段'].value_counts().sort_index()
for age_group, count in age_group_stats.items():
    print(f"  {age_group}: {count}人")

# 5.5 学期统计
print("\n5. 学期统计:")
term_counts = data['学期'].value_counts()
for term, count in term_counts.items():
    print(f"  {term}: {count}人")

# 6. 高级分析
print("\n=== 高级分析 ===")

# 6.1 关联性分析
print("1. 性别与政治面貌关联性:")
gender_policy = pd.crosstab(data['性别'], data['政治面貌'], normalize='index') * 100
print(gender_policy.round(1))

# 6.2 住宿与离校关联
print("\n2. 住宿与离校关联:")
if '是否住宿' in data.columns and '是否离校' in data.columns:
    accom_leave = pd.crosstab(data['是否住宿'], data['是否离校'])
    print(accom_leave)

# 6.3 班级类型分析
print("\n3. 班级类型识别:")
data['班级类型'] = data['班级名称'].apply(lambda x:
                                          'IB班' if 'IB' in str(x) else
                                          '实验班' if '实验' in str(x) else
                                          '普通班' if '普通' in str(x) else
                                          '未分班' if '未分班' in str(x) else
                                          '常规班'
                                          )
class_type_counts = data['班级类型'].value_counts()
for class_type, count in class_type_counts.items():
    print(f"  {class_type}: {count}人")

# 6.4 籍贯地理位置分析
print("\n4. 籍贯省份分布:")


def extract_province(place):
    if pd.isnull(place):
        return '未知'
    place_str = str(place)
    # 常见省份关键词
    provinces = ['浙江', '江苏', '上海', '安徽', '福建', '江西',
                 '山东', '河南', '湖北', '湖南', '广东', '广西',
                 '海南', '四川', '贵州', '云南', '陕西', '甘肃',
                 '青海', '宁夏', '新疆', '黑龙江', '吉林', '辽宁',
                 '内蒙古', '河北', '山西', '天津', '北京', '重庆']

    for province in provinces:
        if province in place_str:
            return province
    return '其他'


data['省份'] = data['籍贯'].apply(extract_province)
province_counts = data['省份'].value_counts().head(10)
for province, count in province_counts.items():
    print(f"  {province}: {count}人")

# 7. 创建更多可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 7.1 班级类型分布
ax1 = axes[0, 0]
class_type_counts = data['班级类型'].value_counts()
colors1 = plt.cm.Set3(np.linspace(0, 1, len(class_type_counts)))
ax1.pie(class_type_counts.values, labels=class_type_counts.index,
        autopct='%1.1f%%', colors=colors1, startangle=90)
ax1.set_title('班级类型分布')

# 7.2 省份分布
ax2 = axes[0, 1]
province_counts = data['省份'].value_counts().head(10)
bars = ax2.barh(province_counts.index, province_counts.values, color='lightblue')
ax2.set_xlabel('人数')
ax2.set_title('生源省份TOP10')
for i, v in enumerate(province_counts.values):
    ax2.text(v + 0.5, i, str(v), va='center')

# 7.3 年龄与政治面貌关系
ax3 = axes[1, 0]
if '年龄' in data.columns and '政治面貌' in data.columns:
    age_policy_data = data[['年龄', '政治面貌']].dropna()
    for policy in age_policy_data['政治面貌'].unique():
        subset = age_policy_data[age_policy_data['政治面貌'] == policy]
        ax3.hist(subset['年龄'], alpha=0.5, label=policy, bins=20)
    ax3.set_xlabel('年龄')
    ax3.set_ylabel('人数')
    ax3.set_title('不同政治面貌的年龄分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# 7.4 性别与住宿情况
ax4 = axes[1, 1]
if '性别' in data.columns and '是否住宿' in data.columns:
    gender_accom = pd.crosstab(data['性别'], data['是否住宿'])
    gender_accom.plot(kind='bar', ax=ax4, color=['lightblue', 'lightcoral'])
    ax4.set_title('性别与住宿情况')
    ax4.set_ylabel('人数')
    ax4.legend(title='是否住宿')

plt.tight_layout()
plt.show()

# 8. 生成统计报告
print("\n" + "=" * 50)
print("学生信息分析报告".center(50))
print("=" * 50)

print(f"\n📊 数据概览:")
print(f"  总记录数: {len(data)} 条")
print(f"  总班级数: {data['班级名称'].nunique()} 个")
print(f"  男生人数: {data[data['性别'] == '男'].shape[0]} 人")
print(f"  女生人数: {data[data['性别'] == '女'].shape[0]} 人")

print(f"\n🎂 年龄特征:")
if '年龄' in data.columns:
    print(f"  平均年龄: {data['年龄'].mean():.1f} 岁")
    print(f"  最小年龄: {data['年龄'].min():.0f} 岁")
    print(f"  最大年龄: {data['年龄'].max():.0f} 岁")
    print(f"  主要年龄段: {age_group_stats.idxmax()} ({age_group_stats.max()}人)")

print(f"\n🎓 政治面貌:")
for policy, stats in policy_stats.iterrows():
    print(f"  {policy}: {int(stats['总人数'])}人 ({stats['总人数'] / len(data) * 100:.1f}%)")

print(f"\n🏠 住宿情况:")
if '是否住宿' in data.columns:
    accom_ratio = data['是否住宿'].value_counts(normalize=True)[1.0] * 100
    print(f"  住宿比例: {accom_ratio:.1f}%")

print(f"\n📍 生源分析:")
print(f"  主要生源省份: {province_counts.index[0]} ({province_counts.iloc[0]}人)")
print(f"  生源多样性: {data['省份'].nunique()} 个省份")

print(f"\n🏫 班级情况:")
print(f"  最大班级: {class_stats['总人数'].idxmax()} ({class_stats['总人数'].max()}人)")
print(f"  最小班级: {class_stats['总人数'].idxmin()} ({class_stats['总人数'].min()}人)")

print(f"\n📅 学期分布:")
for term, count in term_counts.items():
    print(f"  {term}: {count}人 ({count / len(data) * 100:.1f}%)")

print("\n" + "=" * 50)
print("分析完成！".center(50))
print("=" * 50)

# 9. 保存分析结果
print("\n正在保存分析结果...")
data.to_csv('学生信息_分析结果.csv', index=False, encoding='utf-8-sig')
print("分析结果已保存到: 学生信息_分析结果.csv")

# 10. 可选：生成详细的Excel报告
try:
    with pd.ExcelWriter('学生信息_详细报告.xlsx', engine='openpyxl') as writer:
        # 原始数据
        data.to_excel(writer, sheet_name='原始数据', index=False)

        # 班级统计
        class_stats.to_excel(writer, sheet_name='班级统计')

        # 政治面貌统计
        policy_stats.to_excel(writer, sheet_name='政治面貌统计')

        # 省份统计
        province_stats = data['省份'].value_counts().reset_index()
        province_stats.columns = ['省份', '人数']
        province_stats.to_excel(writer, sheet_name='省份统计', index=False)

        # 年龄分段统计
        age_segment_stats = data['年龄分段'].value_counts().reset_index()
        age_segment_stats.columns = ['年龄分段', '人数']
        age_segment_stats.to_excel(writer, sheet_name='年龄分段', index=False)

    print("详细报告已保存到: 学生信息_详细报告.xlsx")
except Exception as e:
    print(f"保存Excel报告时出错: {e}")

print("\n✅ 分析完成！")