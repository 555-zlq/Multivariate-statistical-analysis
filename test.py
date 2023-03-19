import numpy as np
import pandas as pd
from itertools import islice        # 需先安装 itertools
import matplotlib.pyplot as plt
import seaborn as sns  # 统计数据可视化
from scipy import stats
from sklearn.decomposition import PCA
def get_frequency(filename):
    str0 = ["a", "t", "c" , "g"]
    df_count = pd.DataFrame(columns=str0)		# 用于记录频数
    with open(path_data + 'Art-model-data.txt','r', newline='') as filereader:
        k = 0
        for row in islice(filereader, 1, None):   # islice() 跳过第1行
            if len(row)>2:    # 空行可能有换行符，占一定长度
                k = k+1
                k_count = []
                for item in str0:
                    k_count.append(row.count(item))
                df_count.loc[k] = k_count
    df_RowSum = df_count.sum(axis=1)  		# 按行相加，用于将频数转换为频率
    df_freq = df_count.div(df_RowSum, axis="rows")  # 每行除以对应总和， 按行计算频率
    return df_freq.astype("float").round(4)     	# 转换类型为浮点数后，只保留4位小数

path_data = "/home/carton/workspace/python/Multivariate-statistical-analysis/database/期末考核/"     # 设置 数据路径
file0 = path_data + 'Art-model-data.txt'
df_data = get_frequency(file0)

df_corr = df_data.corr()
df_corr.head().round(4)

sns.heatmap(df_data.corr(),annot=True,fmt=".2f", cmap="coolwarm",annot_kws={ "size":14},mask=(df_data.corr()==1))

W,V = np.linalg.eig(df_corr) # 计算特征值和特征向量
sort_id = np.argsort(W)[::-1] # 升序排序后，逆转
W = W[sort_id]; V = V[:,sort_id]
print(W); print(V)
print("特征值之和为：{:.2f}".format(sum(W)))

pca_data = PCA() # 建立模型，用 PCA(n_components=2) 可指定主成分数目
df_tem = df_data.apply(stats.zscore,ddof=1) # 前面进行标准化处理的数据集
principalComponents=pca_data.fit_transform(df_tem)# 训练模型得PCA参数、得分
col_name = ["特征"+str(i) for i in range(1,pca_data.n_components_+1)]
# 自动生成 特征名称
df_res = pd.DataFrame(pca_data.components_.T, columns=col_name, index=df_data.columns[0:]) # 先转置，列为 特征名称, 行为变量名
df_res.loc["特征值",:] = pca_data.n_components_ * pca_data.explained_variance_ / pca_data.explained_variance_.sum()# 归一化处理， (若为协方差矩阵)原始数据 之和 不等于 变量个数
df_res.loc["贡献率",:] = pca_data.explained_variance_ratio_
df_res.loc["累计贡献率",:] = np.cumsum(pca_data.explained_variance_ratio_)

df_pc = pd.DataFrame(data=principalComponents, columns=col_name) # 主成分得分值
df_pc = df_pc.set_index(df_data.index).copy() # 重新设置序号，以原始数据序号为准

# 根据特征1,2 将df_pc分为两类
df_pc["类别"] = df_pc["特征1"].apply(lambda x: "B" if x>0 else "A")
df_pc.head()

# 绘制散点图显示类比的分布
plt.figure(figsize=(8,8))
plt.scatter(df_pc.loc[df_pc["类别"]=="A","特征1"],df_pc.loc[df_pc["类别"]=="A","特征2"],c="r",label="A")

plt.scatter(df_pc.loc[df_pc["类别"]=="B","特征1"],df_pc.loc[df_pc["类别"]=="B","特征2"],c="b",label="B")
plt.legend()
plt.show()

# 用图展示分类结果，并标出每个点的序号
plt.figure(figsize=(8,8))
plt.scatter(df_pc.loc[df_pc["类别"]=="A","特征1"],df_pc.loc[df_pc["类别"]=="A","特征2"],c="r",label="A")
plt.scatter(df_pc.loc[df_pc["类别"]=="B","特征1"],df_pc.loc[df_pc["类别"]=="B","特征2"],c="b",label="B")
for i in range(len(df_pc)):
    plt.text(df_pc.iloc[i,0],df_pc.iloc[i,1],df_pc.index[i],fontsize=10)
plt.legend()
plt.show()












