import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from matplotlib import pyplot as plt
from utils.model import YadavModel
from utils.dataproc import read_img, preprocess_yadav
from utils.eval import top3_as_string

import cv2

FLAGS = flags.FLAGS
flags.DEFINE_string('weights', "./models/gtsrb_usstop/model_best_test", 'path to the weights for the Yadav model')
flags.DEFINE_string('srcimgs', 'bb_images/mcity_iphone_27Aug/grab_mcity_iphone_advstop', 'Path to the images to be classified.')


def main(argv=None):
    imgnames = filter(lambda x: x.lower().endswith(".jpg") or x.lower().endswith(".png"), os.listdir(FLAGS.srcimgs))
    imgs = np.asarray(map(lambda x: preprocess_yadav(x),
                          map(lambda x: cv2.resize(read_img(os.path.join(FLAGS.srcimgs, x)), (FLAGS.img_cols, FLAGS.img_rows)),
                              imgnames))
                      , dtype=np.float32)
    print 'Loaded images from %s'%FLAGS.srcimgs
    sys.stdout.flush()
    results = []
    with tf.Session() as sess:
        model = YadavModel(train=False)
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.weights)
        print 'Loaded model from %s'%FLAGS.weights
        sys.stdout.flush()
        output = sess.run(model.labels_pred, feed_dict={model.features: imgs, model.keep_prob: 1.0})
        # index = np.argmax(output, axis=1)[0]
        index = 5

        print 'index:', index
        heatmap = sess.run(model.heatmap, feed_dict={model.features: imgs, model.keep_prob: 1.0, model.index: index})
        heatmap = np.array(heatmap)
        print 'heatmap.shape1', heatmap.shape

        heatmap = np.sum(np.maximum(heatmap[0], 0), axis=0)
        print 'heatmap.shape1', heatmap.shape
        pool_grad = np.mean(heatmap, axis=(0, 1))
        print 'pool_grad:', pool_grad
        print 'pool_grad.shape', pool_grad.shape
        for i in range(32):
            # heatmap[:, :, i] *= np.mean(heatmap[:, :, i])
            heatmap[:, :, i] *= pool_grad[i]
        heatmap = np.mean(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        #heatmap visualation
        plt.matshow(heatmap)
        plt.savefig("heatmap.png")

        ori_img = cv2.imread("/home/yebin/work/gtsrb-cnn-attack/heat_map_test/6.png")
        ori_img = cv2.resize(ori_img, (256, 256))
        heatmap = cv2.resize(heatmap, (256, 256))

        heatmap = np.uint8(255 * heatmap)
        cv2.imwrite("gray.png", heatmap)
        # cv2.imwrite("heatmap256.png", heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite("heatmap256.png", heatmap)
        superimposed_img = heatmap * 0.4 + ori_img
        cv2.imwrite("heatmap_and_img.png", superimposed_img)
        for i in range(len(imgs)):
            results.append((imgnames[i], top3_as_string(output, i)))

    for i in range(len(results)):
        print results[i][0], results[i][1]

if __name__ == "__main__":
    app.run()