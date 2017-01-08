
# coding: utf-8

# # Set up

# In[ ]:

import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg 
import random

from keras.models import Sequential, json
from keras.layers import Dense, Input, normalization, Conv2D, Flatten, Dropout, ELU
from keras.optimizers import Adam


# In[ ]:

#read the csv file
data_files = pd.read_csv("data/data/driving_log.csv")


# In[ ]:

#helper function for preprocessing image
def preprocess_image(image):
    #cut the top 35 lines to eliminate trees & sky, and bottom 20 lines for the hood
    image = image[35:140,:]
    #resize the images to (66,200) to fit the Nvidia model
    image = cv2.resize(image,(200,66),interpolation=cv2.INTER_AREA)    
    #switch to YUV space to fit the Nvidia model
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


# In[ ]:

#helper function for horizontal transit
def h_trans(image,steer,trans_range):
    rows,cols = (160,320)   
    tr_x = trans_range*(np.random.uniform()-0.5)
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))    
    return image_tr,steer_ang


# In[ ]:

#helper function for data augmentation 
def preprocess_train_data(line_data):
    i_lcr = np.random.randint(3)
    if (i_lcr == 0):
        path_file = 'data/data/'+line_data['left'][0].strip()
        #shift_ang_factor = 1.1
        shift_ang = .25
    if (i_lcr == 1):
        path_file = 'data/data/'+line_data['center'][0].strip()
        shift_ang = 0.
    if (i_lcr == 2):
        path_file = 'data/data/'+line_data['right'][0].strip()
        shift_ang = -.25
    y_steer = line_data['steering'][0] + shift_ang
    
    image = mpimg.imread(path_file)
    image,y_steer = h_trans(image,y_steer,100)
    image = np.array(preprocess_image(image))
    
    i_flip = np.random.randint(2)
    if i_flip==0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    
    return image,y_steer


# In[ ]:

def generate_train_batch(data,batch_size):   
    batch_images = np.zeros((batch_size, 66, 200, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            keep_pr = 0
            #discard steering values that are too small
            while keep_pr == 0:
                x,y = preprocess_train_data(line_data)
                pr_unif = np.random
                if abs(y)<.1:    
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1

            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering


# In[ ]:

def generate_validation_batch(data,batch_size):
    batch_images = np.zeros((batch_size, 66, 200, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_data = data.iloc[[i_line]].reset_index()
            path_file = 'data/data/'+line_data['center'][0].strip()
            x = mpimg.imread(path_file)
            x = np.array(preprocess_image(x))
            y = line_data['steering'][0]           
            #x,y = preprocess_train_data(line_data)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering


# # Network Architecture

# In[ ]:

#Nvidia model
model = Sequential()
model.add(normalization.BatchNormalization(input_shape=(66,200,3), axis=1))
model.add(Conv2D(24, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(36, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(48, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Conv2D(64, 3, 3,))
model.add(ELU())
model.add(Conv2D(64, 3, 3))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(ELU())
model.add(Dense(10))
model.add(Dense(1))
model.summary()


# # Training

# In[ ]:

nb_val = len(data_files)
pr_threshold = 0.75
batch_size = 256

train_generator = generate_train_batch(data_files,batch_size)
valid_generator = generate_validation_batch(data_files,batch_size)


# In[ ]:

model.compile(loss='mse', optimizer=Adam(lr=0.0001),)
history = model.fit_generator(train_generator, samples_per_epoch=20224, 
                              nb_epoch=9,validation_data=valid_generator, nb_val_samples=nb_val)


# # Save architecture & weights

# In[ ]:

# serialize model to JSON
json_string = model.to_json()
with open('model.json','w') as outfile:
    json.dump(json_string,outfile)
# serialize weights to HDF5
model.save_weights("model.h5", overwrite = True)
print("Saved model to disk")

