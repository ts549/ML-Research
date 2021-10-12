#Side project done in May 2021
#Used research done on machine learning concepts to createa  sequential model for public dataset of boston housing prices

#Code block 1
%tensorflow_version 2.x

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow import keras

#Code block 2
(x_train, train_y), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=113)

#Code block 3
feature_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df_train = pd.DataFrame(x_train, columns=feature_columns)
df_eval = pd.DataFrame(x_test, columns=feature_columns)
y_train = pd.DataFrame(train_y)
y_eval = pd.DataFrame(y_test)

normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(x_train)

#Code block 4
model = tf.keras.Sequential([normalizer, tf.keras.layers.Dense(13), tf.keras.layers.Dense(1)])

model.compile(optimizer=tf.optimizers.SGD(), loss = tf.losses.MeanSquaredError())

model.fit(df_train, y_train, epochs=10)

#Code block 5
model.evaluate(df_eval, y_eval)

#Code block 6
print(y_test[0])

#Code block 7
input = np.array([[18.0846, 0., 18.1, 0., 0.679, 6.434, 100., 1.8347, 24., 666., 20.2, 27.25, 29.05]], dtype=np.float32)
result = model.predict(input)
print(result)
