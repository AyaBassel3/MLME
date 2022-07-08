import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.dataset_DNN2 import read_dataset,flatten,plot25
from tensorflow.keras.layers import RandomRotation
import cv2 as cv2
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomHeight, RandomWidth, RandomZoom
from tensorflow.keras.layers import RandomTranslation
#from keras.utils import plot_model
checkpoint_filepath = './dnn_mnist9'
callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
checkpoint_filepath2 = './dnnwithtransferlearning9'
callback2 = ModelCheckpoint(
    filepath=checkpoint_filepath2,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

dim=28

# load dataset
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical
mnist = tf.keras.datasets.mnist
(x_train1, y_train), (x_test, y_test) = mnist.load_data()

x_train1, x_test = np.array(x_train1, np.float32), np.array(x_test, np.float32)

x_train1, x_test =  x_train1 / 255 , x_test / 255
x_train=x_train1





y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)




test_data,test_labels=read_dataset('test',dim)
train_data,train_labels=read_dataset('train',dim)
test_data=test_data.reshape(-1,dim,dim,1)
train_data=train_data.reshape(-1,dim,dim,1) 
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



model.fit(x_train, y_train, epochs=200, batch_size=batch_size,callbacks=[callback],validation_data=(train_data,train_labels))
model.load_weights(checkpoint_filepath)
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy for mnist: %.1f%%" % (100.0 * acc))
loss, acc = model.evaluate(train_data, train_labels, batch_size=batch_size)
print("\nTest accuracy for mnist model on centered train dataset: %.1f%%" % (100.0 * acc))
loss, acc = model.evaluate(test_data, test_labels, batch_size=batch_size)
print("\nTest accuracy for mnist model on centered test dataset: %.1f%%" % (100.0 * acc))
#transfer learning with learning rate scheduler.
model.fit(train_data, train_labels,callbacks=[callback2], epochs=300, batch_size=batch_size,validation_data=(test_data,test_labels))

model.load_weights(checkpoint_filepath2)


scores=model.evaluate(test_data,test_labels)

loss, acc = model.evaluate(test_data, test_labels, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


loss2, acc2 = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy on MNIST test set: %.1f%%" % (100.0 * acc2))