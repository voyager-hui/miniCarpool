import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

place_loc = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                      [4.2, 1], [4.7, 2], [5.3, 3], [5.9, 4], [6.5, 5], [7.3, 6], [8, 7]])
connect = np.loadtxt('./connect.csv', delimiter=',', dtype=int)


def read_log(file_name):
    file = open(file_name)
    episode = -1
    record = []  # episode, up_loc, down_loc, curr_loc
    while 1:
        info = str(file.readline())
        if info[: 3] == 'end':
            break
        elif info[: 7] == 'episode':
            episode += 1
        else:
            s = info.split()
            record.append((episode, int(s[0]), int(s[1]), int(s[2])))
    plot_env(record)


# 绘制当前环境
def plot_env(record):
    # 获取轮次、出租车位置坐标序列
    ep, up_x, up_y, do_x, do_y, x1, y1 = [], [], [], [], [], [], []
    for i in range(len(record)):
        ep.append(record[i][0])
        up_x.append(place_loc[record[i][1], 0])
        up_y.append(place_loc[record[i][1], 1])
        do_x.append(place_loc[record[i][2], 0])
        do_y.append(place_loc[record[i][2], 1])
        x1.append(place_loc[record[i][3], 0])
        y1.append(place_loc[record[i][3], 1])
    ep, up_x, up_y, do_x, do_y, x1, y1 = np.array(ep), np.array(up_x), np.array(up_y), np.array(do_x), np.array(do_y), np.array(x1), np.array(y1)

    # 绘制背景
    fig, ax = plt.subplots()
    plt.axis('off')
    for i in range(16):
        plt.scatter(place_loc[i, 0], place_loc[i, 1], s=10, c='k')
    for i in range(16):
        for j in range(16):
            if connect[i, j] == 0 or j <= i:
                continue
            plt.plot(place_loc[[i, j], 0], place_loc[[i, j], 1], c='lightgray', linewidth=1.0, linestyle="-")

    # 上车点、下车点、当前位置的运动
    def update_points(num):
        t = str(ep[num])
        ax.set_title("Episode No. "+t)
        point_ani1.set_data(up_x[num], up_y[num])
        point_ani2.set_data(do_x[num], do_y[num])
        point_ani.set_data(x1[num], y1[num])
        return point_ani1, point_ani2, point_ani,

    point_ani1, = plt.plot(up_x[0], up_y[0], "g^", linewidth=6)
    point_ani2, = plt.plot(do_x[0], do_y[0], "rv", linewidth=6)
    point_ani, = plt.plot(x1[0], y1[0], "bs", linewidth=15)

    ani = animation.FuncAnimation(fig, update_points, frames=x1.shape[0], interval=10)
    ani.save('video.mp4', fps=40, extra_args=['-vcodec', 'libx264'])
    plt.show()


if __name__ == "__main__":
    read_log('log.txt')
