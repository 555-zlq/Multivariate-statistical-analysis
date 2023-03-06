import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 统计数据可视化
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
# 绘图风格： style= ["darkgrid"，"whitegrid"，"dark"，"white"，"ticks"]，默认darkgrid
sns.set_style(style="darkgrid")
# 颜色风格： themes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
sns.set_palette(palette='bright')
ChinaFonts = {"黑体": "simhei", "宋体": "simsun", "华文楷体": "STKAITI"}
plt.rcParams["font.sans-serif"] = ChinaFonts["黑体"]  # 解决中文乱码问题
plt.rcParams["axes.unicode_minus"] = False  # 解决负号无法正常显示的问题
path_data = "/home/carton/workspace/python/Multivariate-statistical-analysis/database/期末考核/"  # 设置 数据路径
path_pic = "/home/carton/workspace/python/Multivariate-statistical-analysis/save-database/期末考核/"  # 设置 图像保存路径

file_path0 = path_data + "上证50_20230113.xlsx"
df_data0 = pd.read_excel(file_path0)

file_path1 = path_data + "上证50年底交易数据.xlsx"
df_data1 = pd.read_excel(file_path1)

# 1. 请分别计算上证50月度收益率, 对df_data0的每一行在df_data1中找到第一行和最后一行对应的数据，计算收益率
# 枚举df_data0的每一行
for i, row in df_data0.iterrows():
    # 去除row[0]中的括号及括号内的内容
    row[0] = row[0].split("(")[0]
    df_data0.loc[i, "月收益率"] = df_data1.loc[21, row[0]] - df_data1.loc[0, row[0]]

# 按照收益率从高到低排序并画图展示
df_data0.sort_values(by="月收益率", ascending=True, inplace=True)
plt.figure(figsize=(14, 10))
plt.barh(df_data0["上证50"], df_data0["月收益率"])
plt.title("上证50月度收益率")
plt.xlabel("上证50")
plt.ylabel("月收益率")
# plt.savefig(path_pic + "上证50月度收益率.png")
# plt.show()

# 绘制贵州茅台的收益率曲线
plt.figure(figsize=(10, 6))
plt.plot(df_data1["日期"], df_data1["贵州茅台"])
plt.title("贵州茅台的收益率曲线")
plt.xlabel("时间")
plt.ylabel("收益率")
# plt.show()

# 绘制华友钴业的收益率曲线
plt.figure(figsize=(10, 6))
plt.plot(df_data1["日期"], df_data1["华友钴业"])
plt.title("华友钴业的收益率曲线")
plt.xlabel("时间")
plt.ylabel("收益率")
# plt.show()

# 绘制农业银行的收益率曲线
plt.figure(figsize=(10, 6))
plt.plot(df_data1["日期"], df_data1["农业银行"])
plt.title("农业银行的收益率曲线")
plt.xlabel("时间")
plt.ylabel("收益率")
# plt.show()

# 构造一个新的DataFrame，记录每只股票每天的股价, 第一列为企业名称，第二列之后为日期
df_data2 = pd.DataFrame(columns=["企业名称"])
df_data2["企业名称"] = df_data1.columns[1:]
for i in range(22):
    df_data2[df_data1["日期"][i]] = df_data1.iloc[i, 1:].values

rawdata = df_data2
pd.set_option('display.max_columns', None)  # 显示所有的列
pd.set_option('display.max_rows', None)  # 显示所有的行
pd.set_option('display.width', 200)  # 为工不换行显示字段，增大横向显示宽度，。设置横向最大显示200字符


x = rawdata.drop(['企业名称'], axis=1)  # 聚类需要所有变量为数值，因此删掉企业名
kmeans = KMeans(n_clusters=5)  # 初始化KMeans类
kmeans.fit(x)  # 训练模型
rawdata['cluster'] = kmeans.predict(x)  # 计算每个记录所属的簇

# 聚类结果可视化

scaled_x = scale(x)  # 降维前数据标准化，降维效果更好
# PCA降维
pca = PCA(n_components=2)  # 指定降维后的维度为2维
pca.fit(scaled_x)  # 训练PCA
tr = pca.transform(scaled_x)  # 将数据转换为2维

dataset = pd.concat([pd.DataFrame(tr), rawdata['cluster']],axis=1)

c0 = dataset[dataset['cluster'] == 0]  # 筛选出簇类别为0的记录
c1 = dataset[dataset['cluster'] == 1]
c2 = dataset[dataset['cluster'] == 2]
c3 = dataset[dataset['cluster'] == 3]
c4 = dataset[dataset['cluster'] == 4]


plt.plot(c0[0],c0[1],'r.',c1[0],c1[1],'g.',c2[0],c2[1],'b.', c3[0],c3[1],'c.', c4[0],c4[1],'w.')#将3个簇绘制到一-张图上。


# 通过每个簇的索引值，找到对应的企业名称
c0_name = rawdata[rawdata['cluster'] == 0]['企业名称']
c1_name = rawdata[rawdata['cluster'] == 1]['企业名称']
c2_name = rawdata[rawdata['cluster'] == 2]['企业名称']
c3_name = rawdata[rawdata['cluster'] == 3]['企业名称']
c4_name = rawdata[rawdata['cluster'] == 4]['企业名称']



plt.show()