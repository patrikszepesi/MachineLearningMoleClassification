import os
#import cv2
#import keras
import numpy as np
#import pandas as pd
import random as rn
from PIL import Image
#from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
#from IPython.display import SVG
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.python.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
import hypertune

#I ENDED UP NOT USING THE VAL , BUT I STILL UPLOAD IT TO THE BUCKET, I DIDNT WANT TO EXPERIMENT ANYMORE

import efficientnet.keras as efn

from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, save_img
from tensorflow.python.keras.layers import Dense, Flatten,MaxPooling2D, GlobalAveragePooling2D,BatchNormalization,Dropout,Conv2D,MaxPool2D
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from keras.utils.vis_utils import plot_model

from DataHandler import download_data_to_local_directory, upload_data_to_bucket


from tensorflow.python.client import device_lib
import argparse
import shutil
from datetime import datetime

print("tf is running on devices")
print(device_lib.list_local_devices())




def build_model():


    model = Sequential()

    pretrained_model = efn.EfficientNetB0(include_top = False,input_shape = (224, 224, 3),pooling='avg',classes=2, weights = 'imagenet')
    print(len(pretrained_model.layers))

# if we want to set the first  layers of the network to be non-trainable
#for layer in pretrained_model.layers[:len(pretrained_model.layers)-2]:
   # layer.trainable=False
#for layer in pretrained_model.layers[len(pretrained_model.layers)-2:]:
    #layer.trainable=True
#But train Batch Normalization layers
    for layer in pretrained_model.layers:
        if(isinstance(layer,tf.keras.layers.BatchNormalization)):
            layer.trainable=True

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model



def build_data_pipelines(batch_size, train_data_path, val_data_path, eval_data_path):

    img_height= 224
    img_width = 224
    #batch_size= 64#try32 # CHANGE INPUT




    train_datagen = ImageDataGenerator(rescale=1./255,
        zoom_range=0.1,
        validation_split=0.1) # set validation split

    train_generator = train_datagen.flow_from_directory(
        train_data_path ,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    val_generator = train_datagen.flow_from_directory(
        train_data_path , # same directory as training data try val_data_path afterwards
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        ) # set as validation data

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    eval_generator = test_datagen.flow_from_directory(
        eval_data_path,
        target_size=(img_height,img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)


    return train_generator, val_generator, eval_generator


def get_number_of_imgs_inside_folder(directory):

    totalcount = 0

    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            _, ext = os.path.splitext(filename)#_, is place holder bc os.path.splitext splits pathname to roo and extension
            if ext in [".png", ".jpg", ".jpeg"]:
                totalcount = totalcount + 1

    return totalcount


def train(path_to_data, batch_size, epochs,learning_rate,models_bucket_name):

    path_train_data = os.path.join(path_to_data, 'train')
    path_val_data = os.path.join(path_to_data, 'validation')
    path_eval_data = os.path.join(path_to_data, 'test')

    total_train_imgs = get_number_of_imgs_inside_folder(path_train_data)
    total_val_imgs = get_number_of_imgs_inside_folder(path_val_data)
    total_eval_imgs = get_number_of_imgs_inside_folder(path_eval_data)

    print(total_train_imgs, total_val_imgs, total_eval_imgs)

    train_generator, val_generator, eval_generator = build_data_pipelines(
        batch_size=batch_size,
        train_data_path=path_train_data,
        val_data_path=path_val_data,
        eval_data_path=path_eval_data
    )

    classes_dict = train_generator.class_indices

    # model = build_model(nbr_classes=len(classes_dict.keys()))
    model = build_model()

    optimizer = Adam(lr=learning_rate)#1e-5
    red_lr= ReduceLROnPlateau(monitor='val_loss',patience=3,verbose=1,factor=0.7)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    path_to_save_model = './tmp'
    if not os.path.isdir(path_to_save_model):
        os.makedirs(path_to_save_model)

    checkpoint = ModelCheckpoint(
        path_to_save_model,
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        #save_freq='epoch',
        verbose=1
    )



    History = model.fit(     train_generator,
        #steps_per_epoch = total_train_imgs // batch_size,
        validation_data = val_generator,
        #validation_steps = total_val_imgs // batch_size,
        epochs=epochs,
        #verbose=1#,
        callbacks=[early_stopping,checkpoint]
        )

    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=total_train_imgs // batch_size,
    #     validation_data=val_generator,
    #     validation_steps=total_val_imgs // batch_size,
    #     epochs=epochs
    # )


    print("[INFO] Evaluation phase...")


    Y_pred = model.predict(eval_generator)
    y_pred=[]
    for i in range(len(Y_pred)):
        if Y_pred[i][0]>0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print('Confusion Matrix')
    print(metrics.confusion_matrix(eval_generator.classes, y_pred))
    print ()

    print('Classification Report')
    target_names = ['Non nevus', 'Nevus']
    print(metrics.classification_report(eval_generator.classes, y_pred, target_names=target_names))

    print(eval_generator.class_indices)


    print("Starting evaluation using model.evaluate_generator")
    scores = model.evaluate_generator(eval_generator)
    print("Done evaluating!")
    loss = scores[0]
    print(f"loss for hyptertune = {loss}")


    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    zipped_folder_name = f'trained_model_{now}_loss_{loss}'

    shutil.make_archive(zipped_folder_name, 'zip', '/usr/src/app/tmp')

    path_zipped_folder = '/usr/src/app/' + zipped_folder_name + '.zip'
    upload_data_to_bucket(models_bucket_name, path_zipped_folder, zipped_folder_name)



    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(hyperparameter_metric_tag='loss',
                                            metric_value=loss, global_step=epochs)





if __name__ == "__main__":
    # print("start")
    # download_data_to_local_directory('mole-data-bucket', './data')
    # print("end")


    #train(path_to_data, 64, 2)


    parser = argparse.ArgumentParser()

    parser.add_argument("--bucket_name", type=str, help="Bucket name on google cloud storage",
                        default="mole-data-bucket")

    parser.add_argument("--models_bucket_name", type=str, help="Bucket name on google cloud storage for saving trained models",
                        default="trained_models_moles")

    parser.add_argument("--batch_size", type=int, help="Batch size used by the deep learning model",
                        default=32)

    parser.add_argument("--learning_rate", type=float, help="Batch size used by the deep learning model",
                        default=1e-5)
    args = parser.parse_args()

    print("Downloading of data started ...")
    download_data_to_local_directory(args.bucket_name, "./data")
    print("Download finished!")

    path_to_data = './data'
    train(path_to_data, args.batch_size, 35, args.learning_rate, args.models_bucket_name)

    #train(path_to_data,2, 1)
