import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 使用Times New Roman字体
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 
data1_name = 'patient0451_2CH'
data1 = [
    [7380, 7371, 7200, 7338, 6893, 6937, 6696, 6594, 6231, 5784],
    [7614, 7638, 7511, 7479, 7103, 6687, 6969, 6680, 6833, 6375],
]
data2_name = 'patient0457_4CH'
data2 = [
    [9884, 9528, 9167, 8721, 8215, 7937, 7483, 7230, 6974, 6749],
    [10191, 9427, 9072, 8978, 8465, 8137, 7585, 7050, 6983, 6766]
]


data1 = np.array(data1)
data2 = np.array(data2)


# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.plot(data2[0], label='NCM-Net', marker='o')
# ax1.set_title(data1_name)
# ax1.set_xlabel('Time points', fontsize='16', fontweight='bold')
# ax1.set_ylabel('Number of pixel', fontsize='16', fontweight='bold')
# ax1.legend(loc='upper right')
# ax1.tick_params(axis='x', labelsize=12) # 设置第一个子图x轴刻度标签的大小
# ax1.tick_params(axis='y', labelsize=12) # 设置第一个子图y轴刻度标签的大小

# ax2.plot(data2[1], label='ResNet-50 + MA', linestyle='--', marker='*')
# ax2.set_title(data2_name)
# ax2.set_xlabel('Time points', fontsize='16', fontweight='bold')
# ax2.set_ylabel('Number of pixel', fontsize='16', fontweight='bold')
# ax2.legend(loc='upper right')
# ax2.tick_params(axis='x', labelsize=12) # 设置第二个子图x轴刻度标签的大小
# ax2.tick_params(axis='y', labelsize=12) # 设置第二个子图y轴刻度标签的大小


# '''
# 创建一个figure和两个subplots（上下两张图）
fig, (ax1, ax2) = plt.subplots(2, 1)

# 在第一个subplot上画两条折线
ax1.plot(data1[0], label='NCM-Net', marker='o')
ax1.plot(data1[1], label='ResNet-50 + MA', linestyle='--', marker='*')
ax1.set_title(data1_name)
ax1.set_xlabel('Time points', fontsize='16', fontweight='bold')
ax1.set_ylabel('Number of pixel', fontsize='16', fontweight='bold')
ax1.legend(loc='upper right')
ax1.tick_params(axis='x', labelsize=12) # 设置第一个子图x轴刻度标签的大小
ax1.tick_params(axis='y', labelsize=12) # 设置第一个子图y轴刻度标签的大小


# 在第二个subplot上画两条折线
ax2.plot(data2[0], label='NCM-Net', marker='o')
ax2.plot(data2[1], label='ResNet-50 + MA', linestyle='--', marker='*')
ax2.set_title(data2_name)
ax2.set_xlabel('Time points', fontsize='16', fontweight='bold')
ax2.set_ylabel('Number of pixel', fontsize='16', fontweight='bold')
ax2.legend(loc='upper right')
ax2.tick_params(axis='x', labelsize=12) # 设置第二个子图x轴刻度标签的大小
ax2.tick_params(axis='y', labelsize=12) # 设置第二个子图y轴刻度标签的大小
# '''

# 调整布局并显示图表
plt.tight_layout()
plt.savefig(fname='temporal2.png', dpi=300)
# plt.show()
