import numpy as np
import tensorflow as tf
import cv2
import os
from tvnet import *
def calculate(pr_u1,pr_u2,gt_u):
    return np.mean(np.sqrt((pr_u1 - gt_u[:, :, 0])**2 + (pr_u2 - gt_u[:, :, 1])**2))
def loadData(dir):
    img1 = np.zeros((7, 480, 640, 3))
    img2 = np.zeros((7, 480, 640, 3))
    gt_u = np.zeros((7, 480, 640, 2))
    train_dir = "Middlebury/test-data/" + dir
    for i in xrange(7):
        # test = cv2.imread("Middlebury/eval-data/Army/frame07.png")
        img1[i, :] = cv2.resize(cv2.imread(train_dir + "/frame" + str(i + 7).zfill(2) + ".png"), (640, 480))
        img2[i, :] = cv2.resize(cv2.imread(train_dir + "/frame" + str(i + 8).zfill(2) + ".png"), (640, 480))
        # gt_u[i, :] = cv2.optic(train_dir + "/frame" + str(i + 7).zfill(2) + ".flo")
        gt_u[i, :] = readflo(train_dir + "/frame" + str(i + 7).zfill(2) + ".flo")
    return img1, img2, gt_u
def readflo(file_name):
    with open(file_name, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print 'Magic number incorrect. Invalid .flo file'
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file' % (w, h)
            data = np.array(np.fromfile(f, np.float32, count=2 * w * h))
            # Reshape data into 3D array (columns, rows, bands)
            # data2D = np.ndarray.reshape(data, (w, h, 2))
            data2D = data.reshape(int(h), int(w), 2)
            return data2D
def main():
    epe = 0
    x1 = tf.placeholder(shape=[1, 480, 640, 3], dtype=tf.float32)
    x2 = tf.placeholder(shape=[1, 480, 640, 3], dtype=tf.float32)
    tvnet = TVNet()
    u1, u2, rho = tvnet.tvnet_flow(x1, x2)
    sess = tf.Session()#config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)
    saver = tf.train.Saver()
    saver.restore(sess, "ckpt/nn_model.ckpt")
    dir_list = os.listdir("Middlebury/test-data/")
    count = 0 # count the epe
    for i in dir_list:
        train_dir = "Middlebury/test-data/" + i
        # img1, img2, gt = loadData(i)
        for j in xrange(7):
            image1 = cv2.imread(train_dir + "/frame" + str(j + 7).zfill(2) + ".png")
            image2 = cv2.imread(train_dir + "/frame" + str(j + 8).zfill(2) + ".png")
            gt_u = readflo(train_dir + "/frame" + str(j + 7).zfill(2) + ".flo")
            u1_np, u2_np = sess.run([u1, u2], feed_dict={x1: image1[np.newaxis, ...], x2: image2[np.newaxis, ...]})
            epe += calculate(u1_np, u2_np, gt_u)
            count += 1
            # print("epe", epe)
    print(epe/count)
if __name__ == '__main__':
    main()