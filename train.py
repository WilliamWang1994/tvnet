import os
import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio
import time
from tvnet import TVNet, batch_size
max_steps = 20000 #3000
flags = tf.app.flags
# flags.DEFINE_integer("scale", 1, " TVNet scale [3]")
# flags.DEFINE_integer("warp", 1, " TVNet warp [1]")
# flags.DEFINE_integer("iteration", 50, " TVNet iteration [10]")
flags.DEFINE_string("gpu", '0', " gpu to use [0]")
FLAGS = flags.FLAGS
# scale = FLAGS.scale
# warp = FLAGS.warp
# iteration = FLAGS.iteration
if int(FLAGS.gpu > -1):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#
# print 'TVNet Params:\n scale: %d\n warp: %d\n iteration: %d\nUsing gpu: %s' \
#       % (scale, warp, iteration, FLAGS.gpu)

# load image
# img1 = cv2.imread('frame/img1.png')
# img2 = cv2.imread('frame/img2.png')
# print(img1[np.newaxis, ...].shape)
# h, w, c = img1.shape
import random
def loadData(batch_size):
    img1 = np.zeros((batch_size, 384, 512, 3))
    img2 = np.zeros((batch_size, 384, 512, 3))
    for j in range(batch_size):
        i = str(random.randint(1, 22872)).zfill(5)#22872
        img1[j, :] = cv2.imread('../flownet2-tf/data/FlyingChairs_release/data/'+i+'_img1.ppm')
        img2[j, :] = cv2.imread('../flownet2-tf/data/FlyingChairs_release/data/'+i+'_img2.ppm')
        # img_test = img1[np.newaxis, ...]
    return img1, img2
# other_data = os.listdir("Middlebury/other_data/")
def loadMidData():
    eval_data = os.listdir("Middlebury/eval-data/")
    img1 = np.zeros((7, 480, 640, 3))
    img2 = np.zeros((7, 480, 640, 3))
    lod_folder = random.sample(eval_data, 1)[0]
    train_dir ="Middlebury/eval-data/"+lod_folder
    for i in xrange(7):
        # test = cv2.imread("Middlebury/eval-data/Army/frame07.png")
        img1[i, :] = cv2.resize(cv2.imread(train_dir + "/frame" + str(i + 7).zfill(2) + ".png"), (640, 480))
        img2[i, :] = cv2.resize(cv2.imread(train_dir + "/frame" + str(i + 8).zfill(2) + ".png"), (640, 480))
    return img1, img2
# loadMidData()
# model construct
# x1 = tf.placeholder(shape=[batch_size, 480, 640, 3], dtype=tf.float32)
# x2 = tf.placeholder(shape=[batch_size, 480, 640, 3], dtype=tf.float32)
x1 = tf.placeholder(shape=[batch_size, 384, 512, 3], dtype=tf.float32)
x2 = tf.placeholder(shape=[batch_size, 384, 512, 3], dtype=tf.float32)
tf.summary.image('input', [x1, x2])

tvnet = TVNet()
loss, u1, u2 = tvnet.get_loss(x1, x2)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
# init
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8), allow_soft_placement=True)
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
# train_args = tf.global_variables()
# for i in train_args:
#     print(i)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs', sess.graph)
sess.run(tf.global_variables_initializer())

# tf.train.start_queue_runners()

saver = tf.train.Saver(tf.global_variables())
# run model
# u1_np, u2_np = sess.run([train_op, u1, u2], feed_dict={x1: img1[np.newaxis, ...], x2: img2[np.newaxis, ...]})
# min_loss = 10
for step in range(max_steps):
    start_time = time.time()
    # img1, img2 = loadMidData()
    img1, img2 = loadData(batch_size)
    _, loss_value = sess.run([train_op, loss], feed_dict={x1: img1, x2: img2})
    # run_metadata = tf.RunMetadata()
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # summary, _ = sess.run([merged, train_op],
    #                       feed_dict={x1: img1, x2: img2},
    #                       options=run_options,
    #                       run_metadata = run_metadata )#run_metadata=tf.RunMetadata()
    # summary_writer.add_summary(summary)
    # summary_writer.close()               # if loss_value < 0.24:
    #     break
    duration = time.time() - start_time


    if step % 5 == 0:
        # run_metadata = tf.RunMetadata()
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # summary, _ = sess.run([merged, train_op],
        #                       feed_dict={x1: img1, x2: img2},
        #                       options=run_options,
        #                       run_metadata=run_metadata )#run_metadata=tf.RunMetadata
        # train_writer.add_run_metadata(run_metadata, "step%03d" % step)
        # train_writer.add_summary(summary, step)
        # saver.save(sess, './log')
        # if loss_value < 0.24:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
        # print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
# print('min_loss:::', min_loss)
# summary = sess.run(merged)   \
# run_metadata = tf.RunMetadata()
# summary_writer.add_run_metadata(run_metadata)
train_writer.close()
checkpoint_path = os.path.join("ckpt/", "nn_model.ckpt")
saver.save(sess, checkpoint_path)
tf.train.write_graph(sess.graph_def, "model/", "nn_model.pbtxt", as_text=True)

# u1_np = np.squeeze(u1_np)
# u2_np = np.squeeze(u2_np)
# flow_mat = np.zeros([h, w, 2])
# flow_mat[:, :, 0] = u1_np
# flow_mat[:, :, 1] = u2_np

# if not os.path.exists('result'):
#     os.mkdir('result')
# res_path = os.path.join('result', 'result.mat')
# sio.savemat(res_path, {'flow': flow_mat})
