import tensorflow as tf
import tensorflow_hub as hub
#rom tensorflow.contrib import lite
import keras
from keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense,Flatten,Dropout
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
print(X_train[0])
#X_train=np.reshape(X_train,(229,229,3))
#X_test=np.reshape(X_test,(229,229,3))

#Model building
# training parameters
batch_size = 64
optimizer = "rmsprop"



pre_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',  input_shape=(224,224,3))
for layer in pre_model.layers:
  layer.trainable=False

from tensorflow.keras.optimizers import RMSprop

x= layers.Flatten()(pre_model.output)
x=layers.Dense(1024, activation='relu')(x)
x=layers.Dropout(0.2)(x)
x=layers.Dense(2,activation='softmax')(x)

model=tf.keras.Model(pre_model.input,x)
model.compile(optimizer=RMSprop, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=5)
test_loss = model.evaluate(X_test, y_test)
'''export_dir="saved_1"
tf.saved_model.save(model,export_dir)

converter=tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model= converter.convert()
tflite_model_file="Cancer3.tflite"
with open(tflite_model_file,'wb')as f:
  f.write(tflite_model)



#test_loss = model.evaluate(test_images, test_labels)
#print(y_train)

model.save("final_mo.model")
model.predict(X_test[0])'''''