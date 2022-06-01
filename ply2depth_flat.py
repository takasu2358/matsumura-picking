from cgitb import small
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import time

num = 3
WINDOW_WIDTH = 420
WINDOW_HEIGHT = 330
MAX_DEPTH = 1300
MIN_DEPTH = 800
ROTATE180 = 1
left = -90
right = 330
up = 150
bottom = -180
high = 1150
low = 800

def pointcloud_process(pcd_matrix):
    pcd_matrix2 = pcd_matrix[(pcd_matrix[:,0]>left)&(pcd_matrix[:,0]< right) & (pcd_matrix[:, 1]>bottom)&(pcd_matrix[:, 1]<up) & (pcd_matrix[:, 2]>low) & (pcd_matrix[:, 2]<high)]
    pcd_matrix2[:, 0] += -left
    pcd_matrix2[:, 1] += -bottom

    depth = xyz2matrix(pcd_matrix2)

    return depth

def PtoD(xyz):
    depth = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
    x = [[] for i in range(0, len(xyz))]
    y = [[] for i in range(0, len(xyz))]
    z = [[] for i in range(0, len(xyz))]

    for i in range(0, len(xyz)):
        x[i] = int(xyz[i][1])
        y[i] = int(xyz[i][0])
        z[i] = xyz[i][2]

    for i in range(0, len(xyz)):
        if z[i] < MIN_DEPTH:
            depth[x[i]][y[i]] = 255
        if z[i] > MAX_DEPTH:
            depth[x[i]][y[i]] = 0
        if z[i] >= MIN_DEPTH and z[i] <= MAX_DEPTH:
            pixel = 255 * (1.0 - (float(z[i] - MIN_DEPTH) / float(MAX_DEPTH - MIN_DEPTH)))
            depth[x[i]][y[i]] = pixel
    
    return depth

def xyz2matrix(matrix):
    depth = np.zeros((WINDOW_HEIGHT*num, WINDOW_WIDTH*num))
    for xyz in matrix:
        x = int(xyz[1]*num)
        y = int(xyz[0]*num)
        depth[x][y] = high - xyz[2]
    depth = max_pooling(depth)
    return depth

def max_pooling(depth):
    small_depth = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
    for x in range(WINDOW_HEIGHT):
        for y in range(WINDOW_WIDTH):
            kernel = depth[x*3:x*3+3, y*3:y*3+3]
            max_num = np.max(kernel)
            small_depth[x][y] = max_num
    
    return small_depth


def main(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd_matrix = np.asanyarray(pcd.points)

    depth = pointcloud_process(pcd_matrix)

    # depth = PtoD(pcd_matrix2)

    # depth = cv2.rotate(depth, cv2.ROTATE_180)

    # plt.imshow(depth)
    # plt.show()

    return depth

if __name__ == "__main__":
    start = time.time()

    # filepath = "./ply/current_used/u-cylinder1.ply"
    # filepath = "./ply/motion_test/test1.ply"
    # filepath = "./ply/flat.ply"
    filepath = "./input/no-obj.ply"

    depth = main(filepath)

    df  = pd.DataFrame(depth)
    df.to_csv('./input/flat.csv', header=False, index=False)
    # cv2.imwrite('./result/flat.png', depth)

    elapsed_time = time.time() - start#処理の終了時間を取得
    print("実行時間は{}秒でした．".format(elapsed_time))

    plt.imshow(depth)
    plt.show()
