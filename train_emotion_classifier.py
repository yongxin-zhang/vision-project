from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.callbacks import Callback
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import keras
import subprocess
import os

import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.batch_size = 32
config.num_epochs = 20

config.first_layer_conv_width = 3
config.first_layer_conv_height = 3

config.dropout = 0.4 # 0.2 didn't work well to avoid overfit, when 0.4 works!
config.hidden_nodes = 5000 # 200 could only improved val_acc to ~53%, when 2000 made it further to 78%!

input_shape = (48, 48, 1)

def load_fer2013():
    if not os.path.exists("fer2013"):
        print("Downloading the face emotion dataset...")
        subprocess.check_output("curl -SL https://www.dropbox.com/s/opuvvdv3uligypx/fer2013.tar | tar xz", shell=True)
    data = pd.read_csv("fer2013/fer2013.csv")
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.asarray(pixel_sequence.split(' '), dtype=np.uint8).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()

    val_faces = faces[int(len(faces) * 0.8):]
    val_emotions = emotions[int(len(faces) * 0.8):]
    train_faces = faces[:int(len(faces) * 0.8)]
    train_emotions = emotions[:int(len(faces) * 0.8)]
    
    return train_faces, train_emotions, val_faces, val_emotions

# loading dataset

train_faces, train_emotions, val_faces, val_emotions = load_fer2013()
num_samples, num_classes = train_emotions.shape

train_faces /= 255.
val_faces /= 255.

model = Sequential()

# Expr. 1, with conv2D + Maxpooling + dropout0.2, we get ~59% val_acc.
model.add(Conv2D(32,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(48, 48,1),
    activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2))) # Averagepooling didn't add much on val_acc, still around 60%
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,
    (config.first_layer_conv_width, config.first_layer_conv_height),
    input_shape=(48, 48,1),
    activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3)))
#model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten(input_shape=input_shape))
model.add(Dropout(config.dropout))
model.add(Dense(config.hidden_nodes, activation='relu'))
model.add(Dropout(config.dropout))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_faces, train_emotions, batch_size=config.batch_size,
        epochs=config.num_epochs, verbose=1, callbacks=[
            WandbCallback(data_type="image", labels=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
        ], validation_data=(val_faces, val_emotions))


model.save("emotion.h5")



