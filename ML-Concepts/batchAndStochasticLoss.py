#Using Google Colab
#Research done around May 22
#Comments with "Code block n" denote the n-th code cell used in Google Colab

#Code block 1
#Declares tensorflow version
%tensorflow_version 2.x

#Code block 2
#Imports libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Code block 3
#Generates data
batch_size = 20

x_vals = np.random.normal(1, 0.1, 100).astype(np.float32)
y_vals = (x_vals * (np.random.normal(1, 0.05, 100) - 0.5)).astype(np.float32)

def loss_func(y_true, y_pred):
  return tf.reduce_mean(tf.square(y_pred - y_true))

tf.random.set_seed(1)
np.random.seed(0)

#Generates normalized weights and biases
weights = tf.Variable(tf.random.normal(shape=[1]))
biases = tf.Variable(tf.random.normal(shape=[1]))

history_batch = list()
history_stochastic = list()

#Creates the optimizer used for minimizing loss function
my_opt = tf.optimizers.SGD(learning_rate = 0.02)

#Code block 4
#Applies weights and biases to data
def my_output(X, weights, biases):
  return tf.add(tf.multiply(X, weights), biases)

#Code block 5
#Training in batches and loss calculated from batch loss
for i in range(50):
  rand_index = np.random.choice(100, size=batch_size)
  rand_x = [x_vals[rand_index]]
  rand_y = [y_vals[rand_index]]
  with tf.GradientTape() as tape:
    predictions = my_output(rand_x, weights, biases)
    loss = loss_func(rand_y, predictions)
  history_batch.append(loss.numpy())
  gradients = tape.gradient(loss, [weights, biases])
  my_opt.apply_gradients(zip(gradients, [weights, biases]))
  if ((i + 1) % 25 == 0):
    print(f'Biases: {biases.numpy()}')
    print(f'Loss: {loss.numpy()}')

#Data used to calculate stochastic loss
for i in range(50):
  rand_index = np.random.choice(100, size=1)
  rand_x = [x_vals[rand_index]]
  rand_y = [y_vals[rand_index]]
  with tf.GradientTape() as tape:
    predictions = my_output(rand_x, weights, biases)
    loss = loss_func(rand_y, predictions)
  history_stochastic.append(loss.numpy())
  gradients = tape.gradient(loss, [weights, biases])
  my_opt.apply_gradients(zip(gradients, [weights, biases]))
  if ((i + 1) % 25 == 0):
    print(f'Biases: {biases.numpy()}')
    print(f'Loss: {loss.numpy()}')

#Code block 6
#Shows plots of the batch loss and stochastic loss data points
plt.plot(history_batch, 'r--', label = 'Batch loss')
plt.plot(history_stochastic, 'b-', label = 'Stochastic Loss')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()

#Main Learnings
#Batch size affects accuracy of estimating error gradient.
#Batch loss starts at high loss and decreases until it evens out at an asymptote near the value of 0.
#Stochastic loss integrates more randomness into the training so that the optimizer doesn't narrow in on
#local minimums of the loss graph. This may cause inconsistencies in the loss value but it will more likely
#find the global minima.
