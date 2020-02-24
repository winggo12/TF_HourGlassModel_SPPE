import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.platform import gfile
from dataset_prepare import CocoPose

# input_shape = [1, 192, 192, 3]
# output_shape = [1, 48, 48, 14]

with tf.Session() as sess:
    image_path = 'swim2.png'
    model_filename ='model.pb'
    output_node_names = 'hourglass_out_3'

    with tf.gfile.GFile(model_filename, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(restored_graph_def,input_map=None,return_elements=None,name="")

    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image:0")
    output = graph.get_tensor_by_name("%s:0" % output_node_names)

    image_0 = cv2.imread(image_path)
    w, h, _ = image_0.shape
    image_ = cv2.resize(image_0, (192, 192), interpolation=cv2.INTER_AREA)
    with tf.Session() as sess:
        heatmaps = sess.run(output, feed_dict={image: [image_]})
        # np.reshape(image_, [1, input_w_h, input_w_h, 3]),
        CocoPose.display_image(image_,None,heatmaps[0,:,:,:],False)
        # save each heatmaps to disk
        from PIL import Image

        xCoor = []
        yCoor = []

        print(heatmaps.shape[2])

        ##for _ in range(heatmaps.shape[2]):
            ##data = CocoPose.display_image(image_, heatmaps[0,:,:,:], pred_heat=heatmaps[0, :, :, _:(_ + 1)], as_numpy=True)
            ##im = Image.fromarray(data)
            ##im.save("heat_%d.jpg" % _)

        for i in range(14):
            hm_j = heatmaps[0, :, :, i]
            idx = hm_j.argmax()
            y, x = np.unravel_index(idx, hm_j.shape)
            xCoor.append(x)
            yCoor.append(y)

        plt.scatter(xCoor, yCoor)
        plt.gca().invert_yaxis()
        plt.show()
