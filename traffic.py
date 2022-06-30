import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Flatten,Dropout
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D
imgs_path="/home/zorin/Documents/traffic/archive(5)"
df=[]
label=[]
classes=43
print(imgs_path)
for i in range(classes):
    path = os.path.join(imgs_path,'Train', str(i))
    images=os.listdir(path)
    for img in images:
        im = Image.open(path +'/'+ img)
        im=im.resize((30,30))
        im =np.array(im)
        df.append(im)
        label.append(i)
            
df=np.array(df)
label=np.array(label)
print("success")
Path="/home/zorin/Documents/traffic/archive(5)/Test/00002.png"
img =Image.open(Path)
img = img.resize((30,30))
sr = np.array(img)
plt.imshow(img)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,label,test_size=0.2,random_state=42)
print("training shape:",x_train.shape,y_train.shape)
print("testing shape",x_test.shape,y_test.shape)
y_train=to_categorical(y_train,43)
y_test=to_categorical(y_test,43)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu",input_shape=(30,30,3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(43,activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=["accuracy"])
epochs=25
history = model.fit(x_train, y_train, epochs=epochs, batch_size=64, validation_data=(x_test,y_test))
plt.figure(0)
plt.plot(history.history['accuracy'],label="Training accuracy")
plt.plot(history.history['val_accuracy'],label="val accuracy")
plt.title("Accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.plot(history.history['loss'],label="training loss")
plt.plot(history.history['val_loss'],label="val loss")
plt.title("Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
model.save('traffic_classifier.h5')
