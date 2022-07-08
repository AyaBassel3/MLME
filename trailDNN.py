
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.dataset_centralizedandcropedDNN import read_dataset,flatten,plot25
from tensorflow.keras.layers import RandomRotation

from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomHeight, RandomWidth, RandomZoom
from tensorflow.keras.layers import RandomTranslation
#from keras.utils import plot_model

checkpoint_filepath2 = './dnnwithtransferlearning7'

dim=28
test_data,test_labels=read_dataset('test',dim)

test_data=test_data.reshape(-1,dim,dim,1)


# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45
input_size=dim*dim
model = tf.keras.Sequential()
model.add(RandomRotation(0.05))
model.add(RandomTranslation(height_factor=0.05, width_factor=0.05))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(hidden_units, input_dim=input_size))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(hidden_units))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.load_weights(checkpoint_filepath2)
scores=model.evaluate(test_data,test_labels)

loss, acc = model.evaluate(test_data, test_labels, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))