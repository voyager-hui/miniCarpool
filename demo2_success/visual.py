import os
import io
import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 解析log文件的数据，以便后续的可视化
def read_log(file_name="two_taxi_from11to7.txt"):
    file = open(file_name)
    record = []
    while 1:
        info = str(file.readline())
        if info[: 3] == 'end':
            break
        elif info[: 7] == 'episode':
            record.append([])
        else:
            s = info.split()
            record[-1].append((int(s[1]), int(s[3])))
    plot_env(record)


# 绘制当前环境
def plot_env(record):
    place_loc = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                          [4.2, 1], [4.7, 2], [5.3, 3], [5.9, 4], [6.5, 5], [7.3, 6], [8, 7]])
    connect = np.loadtxt('./connect.csv', delimiter=',', dtype=int)

    # 获取出租车位置坐标序列
    # ep = [i for i in range(5000)]
    x1, y1 = [], []
    x2, y2 = [], []
    mylist = [i for i in range(0, 70)]
    for episode in mylist:
        for k in range(len(record[episode])):
            # ep.append(episode)
            x1.append(place_loc[record[episode][k][0], 0])
            y1.append(place_loc[record[episode][k][0], 1])
            x2.append(place_loc[record[episode][k][1], 0])
            y2.append(place_loc[record[episode][k][1], 1])

    # 绘制背景
    # fig = plt.figure(tight_layout=True)
    # ax = plt.subplot(111)
    fig, ax = plt.subplots()
    # 绘制地图上主要地点
    for i in range(16):
        plt.scatter(place_loc[i, 0], place_loc[i, 1], s=10, c='k')
    # 绘制地点之间的路线
    for i in range(16):
        for j in range(16):
            if connect[i, j] == 0 or j <= i:
                continue
            plt.plot(place_loc[[i, j], 0], place_loc[[i, j], 1], c='lightgray', linewidth=1.0, linestyle="-")
    # 绘制地图上上车点和下车点
    plt.scatter(place_loc[13, 0], place_loc[13, 1], s=10, marker='s', c='b')
    plt.text(place_loc[13, 0], place_loc[13, 1], str("up point"), fontsize=8)
    plt.scatter(place_loc[7, 0], place_loc[7, 1], s=10, marker='s', c='b')
    plt.text(place_loc[7, 0], place_loc[7, 1], str("down point"), fontsize=8)

    def update_points(num):
        # t = ep[num]
        ax.set_title("DQN training. Two taxis. (40fps)")
        point_ani.set_data(x1[num], y1[num])
        point_ani.set_data(x2[num], y2[num])
        return point_ani,

    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    point_ani, = plt.plot(x1[0], y1[0], "rs")
    point_ani, = plt.plot(x2[0], y2[0], "ys")

    ani = animation.FuncAnimation(fig, update_points, frames=1000, interval=25)
    # ani.save('carpool.mp4', writer=writer)
    ani.save('video.gif', writer='pillow')
    plt.show()


    '''
    from PIL import Image
    img = cv2.imread('hhh.png')  # 获取图像的尺寸
    video_writer = cv2.VideoWriter('carpool.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (img.shape[1], img.shape[0]))
    mylist = [0]
    my = [i for i in range(960, 1000)]
    mylist.extend(my)
    for episode in mylist:
        for k in range(len(record[episode])):
            # 绘制出租车的位置
            plt.scatter(place_loc[record[episode][k], 0], place_loc[record[episode][k], 1], s=20, marker='s', c='r')

            plt.title('DQN training No. ' + str(episode) + ' episode (30fps)')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            data = np.asarray(Image.open(buf))[:, :, :3]
            video_writer.write(data)
            plt.clf()
    video_writer.release()
    '''


if __name__ == "__main__":
    read_log()
