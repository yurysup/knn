import tensorflow.compat.v1 as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tqdm import tqdm

tf.disable_v2_behavior()

# Load and normalize dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

# k-param
k_max = 30
# number of test samples - to be able to reduce total execution time
n = 5000

# TensorFlow Graph calculation model
x_train_ph = tf.placeholder(tf.float32, shape=x_train.shape)
y_train_ph = tf.placeholder(tf.float32, shape=y_train.shape)
x_test_ph = tf.placeholder(tf.float32, shape=x_test.shape[1:])
# Calculate L1-distances as negative to allow picking first top K entries after DESC sorting
distances = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.abs(tf.subtract(x_train_ph, x_test_ph)), axis=1), axis=1))
# Find top K entries after DESC sorting
top_k_values, top_k_indices = tf.nn.top_k(distances, k=k_max+1, sorted=True)
top_k_max_labels = tf.gather(y_train_ph, top_k_indices)
predictions = []
# Calculate predictions for different k - [1, k_max]
for k in range(1, k_max + 1):
    top_k_labels = tf.slice(top_k_max_labels, begin=[0], size=[k])
    unique_classes, ids, top_k_labels_counts = tf.unique_with_counts(top_k_labels)
    prediction = tf.gather(unique_classes, tf.argmax(top_k_labels_counts))
    predictions.append(prediction)
predictions = tf.stack(predictions)

# Start TensorFlow Session
correct_predictions_nums = np.zeros(k_max)
with tf.Session() as session:
    for i in tqdm(range(0, n)):
        predicted_values = session.run(predictions, feed_dict={x_test_ph:x_test[i], x_train_ph:x_train, y_train_ph:y_train})
        for k in range(0, k_max):
            if int(predicted_values[k]) == y_test[i]:
                correct_predictions_nums[k] += 1
    accuracies = correct_predictions_nums / n * 100

# Plot accuracy-k dependency
x_axis = range(k_max)
plt.plot(x_axis, accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy, %')
plt.show()