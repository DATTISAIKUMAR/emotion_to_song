
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

data=Sequential()
data.add(layers.RandomFlip("horizontal_and_vertical"))
data.add(layers.RandomRotation(30))
data.add(layers.RandomContrast(0.5))
data.add(layers.RandomTranslation(0.1,0.4))
data.add(layers.RandomBrightness(0.2))
data.add(layers.RandomZoom(0.2))

from tensorflow.keras.preprocessing.image import img_to_array,array_to_img,load_img
img=load_img("C:\\Users\\sai\\Desktop\\images\\Fear\\Fear18.jpg")
plt.imshow(img)
arr=img_to_array(img)
arr.shape
arr=arr.reshape((1,)+arr.shape)
arr.shape
plt.imshow(data(img).numpy().astype("uint8"))
from keras.preprocessing.image import ImageDataGenerator
data1=ImageDataGenerator()

def hello(img):
    img=data(img).numpy().astype('uint8')
    arr=img_to_array(img)
    arr=arr.reshape((1,)+arr.shape)
    return arr

i=0
for batch in data1.flow(arr,batch_size=1,save_to_dir="C:\\Users\\sai\\Desktop\\images\\Fear",save_prefix='Fear',save_format='jpg'):
    i+=1
    if(i>=10):
        break
"""


import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential #importing model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten #importing different layers
from json import *



train=ImageDataGenerator(rescale=1/255)
train_data=train.flow_from_directory("images\images",
                                    batch_size=3,
                                    target_size=(250,250))


val_data=train.flow_from_directory("images1\images",
                                    batch_size=3,
                                    target_size=(250,250))
print("training classes :\n")
print(train_data.class_indices)
print("index of training classes:\n")
print(train_data.classes)
print("\n")





print("nework model build:\n")
cnn=Sequential()


#adding cnn(convolution neural network)

cnn.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",kernel_initializer="random_normal",input_shape=(250,250,3)))
cnn.add(MaxPooling2D((3,3)))                                      #find the maximum element from the matrix using maxpooling
cnn.add(Conv2D(filters=64,kernel_size=(3,3),kernel_initializer="random_normal",activation="relu"))
cnn.add(MaxPooling2D((3,3)))
cnn.add(Conv2D(filters=80,kernel_size=(3,3),kernel_initializer="random_normal",activation="relu"))
cnn.add(MaxPooling2D((3,3)))


#converting multi dimensional to a single dimensional array
cnn.add(Flatten())

# applying ann layers
cnn.add(Dense(64,activation="relu"))
cnn.add(Dense(9,activation="softmax"))





cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)




print("train the data :\n")
history=cnn.fit(train_data,validation_data=val_data,epochs=10) #data fit into the model




print("target image shape:\n")
print(train_data[0][0].shape)
print("\n")


print("predict values:\n")
y_prediction=cnn.predict(train_data)   #predicting values of that dataset
print(y_prediction)



print("about model:\n")
cnn.summary()





print("graph between accuracy and Loss:\n")


# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_loss'], label='testing Loss')
plt.plot(history.history['val_accuracy'], label='testing accuracy')
plt.xlabel('Loss')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""
print("input image reshape the size:\n")


string=input()
img=cv2.imread(string)
img1=cv2.resize(img,(250,250))
img1=np.expand_dims(img1,axis=0)
print(img1.shape)
print("\n")
print("predict values of image:\n")
val=cnn.predict(img1)
print(val)
print()
val=np.argmax(val)
print("predict the class index:\n")
print(val)



print()

if(val==0):
  print("Anger")
elif(val==1):
  print("Contempt")
elif(val==2):
  print("Disgust")
elif(val==3):
  print("Fear")
elif(val==4):
  print("Happy")
elif(val==5):
  print("Neutral")
elif(val==6):
  print("Sad")
elif(val==7):
  print("Surprised")



print("\n")


print("image with emotion name:\n")
plt.imshow(img)
plt.show()

"""



model_json=cnn.to_json()
with open("model_architecture.json","w") as json_file:
    json_file.write(model_json)


cnn.save_weights("model_weights.weights.h5")