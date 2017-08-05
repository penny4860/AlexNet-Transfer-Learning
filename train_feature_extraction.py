import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
import pickle
with open('train.p', 'rb') as f:
    dataset = pickle.load(f)
print(dataset.keys())

images = dataset['features']
labels = dataset['labels']

# TODO: Split data into training and validation sets.
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# TODO: Define placeholders and resize operation.
# TODO: pass placeholder as first argument to `AlexNet`.
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
# TODO: Add the final layer for traffic sign classification.

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
from base import _Model, train, evaluate
class FeatureExtFromAlexNet(_Model):
    def _create_input_placeholder(self):
        return tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")

    def _create_inference_op(self):
        resized = tf.image.resize_images(self.X, (227, 227))
        
        # TODO: pass placeholder as first argument to `AlexNet`.
        fc7 = AlexNet(resized, feature_extract=True)
        # NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
        # past this point, keeping the weights before and up to `fc7` frozen.
        # This also makes training faster, less work to do!
        fc7 = tf.stop_gradient(fc7)
        
        # TODO: Add the final layer for traffic sign classification.
        nb_classes = 43
        shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
        
        with tf.name_scope("last_layer"):
            weights = tf.Variable(tf.random_normal(shape, stddev=0.01), name="weights")
            bias = tf.Variable(tf.zeros(nb_classes), name="bias")
            logits = tf.add(tf.matmul(fc7, weights), bias)
        
        return logits

    def _create_loss_op(self):
        one_hot_y = tf.one_hot(self.Y, 43)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.inference_op, labels=one_hot_y))

# TODO: Train and evaluate the feature extraction model.
train(FeatureExtFromAlexNet(),
      X_train, y_train, X_test, y_test, ckpt="ckpt/cnn")
