# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rc

    plt.rc('font', family='SimHei', size=13)
    plt.rcParams['axes.unicode_minus'] = False
    B_Dist = stats.binom(10, 0.5)  # 定义分布
    x = range(11)  # 取值范围，从0到10
    plt.bar(x, B_Dist.pmf(x))  # 绘图，B_Dist.pmf(x) 生成x在该分布下对应的值
    plt.title('二项分布概率质量函数')
    plt.show()
        # 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
