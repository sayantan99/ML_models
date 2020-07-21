import tensorflow as tf
import tensorflow_hub as hub
#rom tensorflow.contrib import lite
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
import os
import numpy as np
from PIL import Image

## Reading the images and concatenate them
# Loading training data
folder_benign_train = "C:\\Users\\Dell\\Downloads\\train\\benign"
folder_malignant_train="C:\\Users\\Dell\\Downloads\\train\\malignant"



read= lambda imname: np.asarray(Image.open(imname).convert('RGB'))

ims_benign=[read(os.path.join(folder_benign_train,filename)) for filename in os.listdir(folder_benign_train)]
ims_malignant=[read(os.path.join(folder_malignant_train,filename)) for filename in os.listdir(folder_malignant_train)]

X_benign= np.array(ims_benign,dtype="uint8")
X_malignant=np.array(ims_malignant,dtype="uint8")
#print(X_benign.shape[0])
#print(X_benign)
#print(X_malignant.shape[0])

# loading testing data
folder_benign_test="C:\\Users\\Dell\\Downloads\\test\\benign"
folder_malignant_test="C:\\Users\\Dell\\Downloads\\test\\malignant"

ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

X_train = np.concatenate((X_benign, X_malignant), axis = 0)
y_train = np.concatenate((y_benign, y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)

# Shuffle data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]

y_train = to_categorical(y_train, num_classes= 2)
y_test = to_categorical(y_test, num_classes= 2)

X_train = X_train/255.
X_test = X_test/255.

print(X_train.shape)
print(X_test.shape)

#Model building
# training parameters
#batch_size = 64
#optimizer = "rmsprop"

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
  tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),

  tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
  tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
  tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units=4096,activation="relu"),
  tf.keras.layers.Dense(units=4096,activation="relu"),
  tf.keras.layers.Dense(units=2, activation="softmax")
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, steps_per_epoch=100,validation_steps=10,epochs=5,batch_size=32)
test_loss = model.evaluate(X_test, y_test)
export_dir="saved_1"
tf.saved_model.save(model,export_dir)


converter=tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model= converter.convert()
tflite_model_file="Cancer3.tflite"
with open(tflite_model_file,'wb')as f:
  f.write(tflite_model)



#test_loss = model.evaluate(test_images, test_labels)'''
#print(y_train)

