import matplotlib.pyplot as plt

# 读取loss.txt
with open("loss.txt", "r") as f:
    loss_list = f.readlines()

# 提取出Loss的值
loss1_list = []
loss2_list = []
loss3_list = []
total_loss_list = []
for loss in loss_list:
    loss = loss.split()
    loss1_list.append(float(loss[0]))
    loss2_list.append(float(loss[1]))
    loss3_list.append(float(loss[2]))
    total_loss_list.append(float(loss[3]))

# 绘制Loss的变化曲线
plt.plot(loss1_list, label="loss1")
plt.plot(loss2_list, label="loss2")
plt.plot(loss3_list, label="loss3")
plt.plot(total_loss_list, label="total_loss")
plt.legend()
plt.show()