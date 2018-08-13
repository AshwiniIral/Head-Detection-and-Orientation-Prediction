# used to predict the direction of a image

import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))
pred_images = sys.argv[1:]

image_size = 64
num_channels = 3

print(' ')

def find_pose(filename):
    images = []
    raw_image = cv2.imread(filename)
    image = cv2.resize(raw_image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)

    height, width = image.shape[:2]
    print(height)
    print(width)

    # height, width = raw_image.shape[:2]
    # print(height)
    # print(width)
    # thumbnail = cv2.resize(raw_image, (width / 10, height / 10), interpolation=cv2.INTER_AREA)
    cv2.imshow('exampleshq', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    ## Restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('face-image.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir('data1'))))

    ## Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]

    # labels = 'Right', 'Left', 'Down', 'Up', 'Front'

    train_path = "data1"
    labels = os.listdir(train_path)

    print(result)

    image_type = (result[0].tolist().index(max(result[0])))
    print(labels[image_type])
    return labels[image_type]


# for path in pred_images:
#     print('Predicting - '+ path )
#     filename = dir_path + '/' + path
#     print(filename)
#     find_pose(filename)
