#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:55:05 2019

@author: andrew
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
basePath = os.path.abspath("/home/andrew/Documents/Machine Learninng/ItraCranialProject/rsna-intracranial-hemorrhage-detection/")
Data = pd.read_csv(basePath+'/stage_1_train.csv')
print(Data.head(10))



splitData = Data['ID'].str.split('_', expand = True)
Data['class'] = splitData[2]
Data['fileName'] = splitData[0] + '_' + splitData[1]
Data = Data.drop(columns=['ID'],axis=1)
del splitData
print(Data.head(10))



Final_Data = Data[['Label', 'fileName', 'class']].drop_duplicates().Final_table(index = 'fileName',columns=['class'], values='Label')
Final_Data = pd.DataFrame(Final_Data.to_records())
print(Final_Data.head(10))




import matplotlib.image as pltimg
import pydicom

# prints 25 * 25 dicom Images
fig = plt.figure(figsize = (20,10))
rows = 5
columns = 5
trainImages = os.listdir(basePath + '/stage_1_train_images/')
for i in range(rows*columns):
    ds = pydicom.dcmread(basePath + '/stage_1_train_images/' + trainImages[i*100+1])
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot    
    
# Prints 25*25 dicom images for each class == 1
colsToPlot = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
rows = 5
columns = 5
for i_col in colsToPlot:
    fig = plt.figure(figsize = (20,10))
    trainImages = list(Final_Data.loc[Final_Data[i_col]==1,'fileName'])
    plt.title(i_col + ' Images')
    for i in range(rows*columns):
        ds = pydicom.dcmread(basePath + '/stage_1_train_images/' + trainImages[i*100+1] +'.dcm')
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)        
        fig.add_subplot    

for i_col in colsToPlot:
    plt.figure()
    ax = sns.countplot(Final_Data[i_col])
    ax.set_title(i_col + ' class count')
   


#dropping of corrupted image from dataset as mentioned in https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/109701#latest-640597
Final_Data = Final_Data.drop(list(Final_Data['fileName']).index('ID_6431af929'))

import keras
from keras.layers import Dense, Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Input,BatchNormalization,AveragePooling2D,LeakyReLU,ZeroPadding2D,Add
from keras.models import Sequential, Model
from keras.initializers import glorot_uniform
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2

Final_Data = Final_Data.sample(frac=1).reset_index(drop=True)
train_df,val_df = train_test_split(Final_Data,test_size = 0.03, random_state = 42)
batch_size = 64


YTrain = train_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
Y_Val = val_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
train_files = list(train_df['fileName'])

def readDCMFile(fileName):
    ds = pydicom.read_file(fileName) # read dicom image
    img = ds.pixel_array # get image array
    img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA) 
    return img

def generateImageDataTrain(train_files,YTrain):
    numBatches = int(np.ceil(len(train_files)/batch_size))
    while True:
        for i in range(numBatches):
            batchFiles = train_files[i*batch_size : (i+1)*batch_size]
            x_batch_data = np.array([readDCMFile(basePath + '/stage_1_train_images/' + i_f +'.dcm') for i_f in tqdm(batchFiles)])
            y_batch_data = YTrain[i*batch_size : (i+1)*batch_size]
            x_batch_data = np.reshape(x_batch_data,(x_batch_data.shape[0],x_batch_data.shape[1],x_batch_data.shape[2],1))            
            yield x_batch_data,y_batch_data
            
def generateTestImageData(test_files):
    numBatches = int(np.ceil(len(test_files)/batch_size))
    while True:
        for i in range(numBatches):
            batchFiles = test_files[i*batch_size : (i+1)*batch_size]
            x_batch_data = np.array([readDCMFile(basePath + '/stage_1_test_images/' + i_f +'.dcm') for i_f in tqdm(batchFiles)])
            x_batch_data = np.reshape(x_batch_data,(x_batch_data.shape[0],x_batch_data.shape[1],x_batch_data.shape[2],1))
            yield x_batch_data  



dataGenerator = generateImageDataTrain(train_files,train_df[colsToPlot])
val_files = list(val_df['fileName'])
X_Val = np.array([readDCMFile(basePath + '/stage_1_train_images/' + i_f +'.dcm') for i_f in tqdm(val_files)])

Y_Val = val_df[colsToPlot]

# loss function definition courtesy https://www.kaggle.com/akensert/resnet50-keras-baseline-model
from keras import backend as K
def logloss(y_true,y_pred):      
    eps = K.epsilon()
    
    class_weights = np.array([2., 1., 1., 1., 1., 1.])
    
    y_pred = K.clip(y_pred, eps, 1.0-eps) # clips the values minimum of epsilon 1e-07 to max 1- epsilon

    #compute logloss function (vectorised)  -(ylog(p) +(1-y) * log(1-p)* W)
    out = -( y_true *K.log(y_pred)*class_weights
            + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights) 
    return K.mean(out, axis=-1)

def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for the this competition
    """
    
    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1) # expands dimensions 
        return K.sum(K.dot(arr, weights), axis=1) / scl # sum x*1/ sum w
    return K.mean(arr, axis=1)

def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """      
    
    eps = K.epsilon()
    
    class_weights = K.variable([2., 1., 1., 1., 1., 1.])
    
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    loss = -(y_true*K.log(y_pred)
            + (1.0 - y_true) * K.log(1.0 - y_pred))
    
    loss_samples = _normalized_weighted_average(loss,class_weights)
    
    return K.mean(loss_samples)

def inceptionKeras(input_img, numFilters11_1, numFilters11_2, numFilters11_3, numFilters33_1, 
                   numFilters55_1,numFilters_pool): 
    tower_11_1 = Conv2D(numFilters11_1, (1,1), padding='same', activation='relu')(input_img)
    tower_11_2 = Conv2D(numFilters11_2, (1,1), padding='same', activation='relu')(input_img)
    tower_33_1 = Conv2D(numFilters33_1, (3,3), padding='same', activation='relu')(tower_11_2) 
    tower_11_3 = Conv2D(numFilters11_3, (1,1), padding='same', activation='relu')(input_img) 
    tower_55_1 = Conv2D(numFilters55_1, (5,5), padding='same', activation='relu')(tower_11_3)
    tower_33_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_33_pool = Conv2D(numFilters_pool, (1,1), padding='same', activation='relu')(tower_33_pool)
    output = keras.layers.concatenate([tower_11_1, tower_33_1, tower_55_1, tower_33_pool], axis = 3)    
    output = Activation('relu')(output)
    return output

input_img = Input(shape=(64,64,1))
layer_1 = Conv2D(filters = 64, kernel_size = (5,5), strides = 1, padding = 'same', activation='relu')(input_img) 
layer_2 = Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'same', activation='relu')(layer_1) 
layer_2 = MaxPooling2D(pool_size = (3,3), padding = 'same', strides = 2)(layer_2) 
layer_2 = BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(layer_2) 
layer_incp_1 = inceptionKeras(layer_2,8,64,8,96,16,8) 
layer_incp_2 = inceptionKeras(layer_incp_1,64,96,16,128,32,32) 
layer_incp_3 = inceptionKeras(layer_incp_2,160,112,24,224,64,64) 
layer_3 = MaxPooling2D(pool_size = (3,3),padding = 'same',strides = 2)(layer_incp_3) 
layer_3 = BatchNormalization()(layer_3) 
layer_incp_4 = inceptionKeras(layer_3,128,128,32,256,64,64) 
layer_4 = AveragePooling2D(pool_size = (7,7),padding = 'same',strides = 7)(layer_incp_4) 
output = Flatten()(layer_4) 
output = Dropout(0.5)(output) 
output = Dense(512,activation='relu')(output) 
out = Dense(6, activation='sigmoid')(output)

def convolutionBlock(X,f,filters,stage,block,s):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    X_shortcut = X
    F1,F2,F3 = filters
    X = Conv2D(filters = F1, kernel_size = (1,1),strides = s, padding = 'valid',name = conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f,f),strides = 1, padding = 'same',name = conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2c')(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1),strides = s, padding = 'valid',name = conv_name_base + '1',
               kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'1')(X_shortcut)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

def identityBlock(X,f,filters,stage,block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    X_shortcut = X
    F1,F2,F3 = filters
    X = Conv2D(filters = F1, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F2, kernel_size = (f,f),strides = 1, padding = 'same',name = conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = F3, kernel_size = (1,1),strides = 1, padding = 'valid',name = conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3,momentum=0.99, epsilon=0.001,name = bn_name_base+'2c')(X)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X

input_img = Input((64,64,1))
X = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(input_img)
X = BatchNormalization(axis=3, name='initial_bn')(X)
X = Activation('relu', name='initial_relu')(X)
X = ZeroPadding2D((3, 3))(X)

# Stage 1
X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
X = BatchNormalization(axis=3, name='bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3, 3), strides=(2, 2))(X)

# Stage 2
X = convolutionBlock(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
X = identityBlock(X, 3, [64, 64, 256], stage=2, block='b')
X = identityBlock(X, 3, [64, 64, 256], stage=2, block='c')

# Stage 3 (≈4 lines)
X = convolutionBlock(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
X = identityBlock(X, 3, [128, 128, 512], stage=3, block='b')
X = identityBlock(X, 3, [128, 128, 512], stage=3, block='c')
X = identityBlock(X, 3, [128, 128, 512], stage=3, block='d')

# Stage 4 (≈4 lines)
X = convolutionBlock(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='b')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='c')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='d')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='e')
X = identityBlock(X, 3, [256, 256, 1024], stage=4, block='f')

# Stage 5 (≈4 lines)
X = convolutionBlock(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
X = identityBlock(X, 3, [512, 512, 2048], stage=5, block='b')
X = identityBlock(X, 3, [512, 512, 2048], stage=5, block='c')


# AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
# output layer
X = Flatten()(X)
out = Dense(6,name='fc' + str(6),activation='sigmoid')(X)

X_Val = np.reshape(X_Val,(X_Val.shape[0],X_Val.shape[1],X_Val.shape[2],1))



model_conv = Model(inputs = input_img, outputs = out)
#model_conv.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
model_conv.compile(optimizer='Adam',loss = logloss,metrics=[weighted_loss])
model_conv.summary()
history_conv = model_conv.fit_generator(dataGenerator,steps_per_epoch=500, epochs=20,validation_data = (X_Val,Y_Val),verbose = False)



testInfo = pd.read_csv(basePath+'/stage_1_sample_submission.csv')
splitData = testInfo['ID'].str.split('_', expand = True)
testInfo['class'] = splitData[2]
testInfo['fileName'] = splitData[0] + '_' + splitData[1]
testInfo = testInfo.drop(columns=['ID'],axis=1)
del splitData
Final_testInfo = testInfo[['fileName', 'class','Label']].drop_duplicates().Final_table(index = 'fileName',columns=['class'], values='Label')
Final_testInfo = pd.DataFrame(Final_testInfo.to_records())
test_files = list(Final_testInfo['fileName'])
testDataGenerator = generateTestImageData(test_files)
temp_pred = model_conv.predict_generator(testDataGenerator,steps = Final_testInfo.shape[0]/batch_size,verbose = True)

submission_df = Final_testInfo
submission_df['any'] = temp_pred[:,0]
submission_df['epidural'] = temp_pred[:,1]
submission_df['intraparenchymal'] = temp_pred[:,2]
submission_df['intraventricular'] = temp_pred[:,3]
submission_df['subarachnoid'] = temp_pred[:,4]
submission_df['subdural'] = temp_pred[:,5]



submission_df = submission_df.melt(id_vars=['fileName'])
submission_df['ID'] = submission_df.fileName + '_' + submission_df.variable
submission_df['Label'] = submission_df['value']
print(submission_df.head(20))

