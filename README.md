# UDACITY_Behavioral_Cloning
Udacity - SDCND - term 1 - P3

I chose 3 candidates for the base-archetecute: 
  1. commaai archetecture: 
     normalization layer -> 
     conv2D((16,8,8),subsample(4,4)) -> 
     conv2D((32,5,5),subsample(2,2)) -> 
     conv2D((64,5,5),subsample(2,2)) -> 
     flatlayer -> 
     fully_connected_layer(512) -> 
     fully_connected_layer(1)

  2. Nvidia archetecture: 
     normalization layer -> 
     conv2D((24,5,5),subsample(2,2)) -> 
     conv2D((36,5,5),subsample(2,2)) -> 
     conv2D((48,5,5),subsample(2,2)) -> 
     conv2D((64,3,3)) -> 
     conv2D((64,3,3)) -> 
     flatlayer -> 
     fully_connected_layer(100) ->
     fully_connected_layer(50) ->
     fully_connected_layer(10) ->
     fully_connected_layer(1) ->

  3. archetecutre published by Vivek Yadav:
     conv2D((3,1,1)) -> 
     conv2D((32,3,3),subsample(2,2)) -> 
     conv2D((64,3,3),subsample(2,2)) ->
     conv2D((128,3,3),subsample(2,2)) -> 
     flatten layer -> 
     fully_connected_layer(512) ->
     fully_connected_layer(64) ->
     fully_connected_layer(16) -> 
     fully_connected_layer(1)

Using training data (8036 center images) provided by Udacity without any preprocessing or data augmentation, I tested the simulation results of the models got from these three DNNs. The best performance came from Nvidia DNN: the Nvidia DNN was the only one that got the car to pass the bridge.

the details of the final archetecture are shown below:

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
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(Dense(1))
model.summary()

Layer (type)                     Output Shape          Param #     Connected to                  
====================================================================================================
batchnormalization_1 (BatchNormal(None, 66, 200, 3)    132         batchnormalization_input_1[0][0] 
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        batchnormalization_1[0][0]       
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 31, 98, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 14, 47, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       elu_2[0][0]                      
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 5, 22, 48)     0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 20, 64)     27712       elu_3[0][0]                      
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 3, 20, 64)     0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 18, 64)     36928       elu_4[0][0]                      
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 1, 18, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1152)          0           elu_5[0][0]                      
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           115300      flatten_1[0][0]                  
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        elu_6[0][0]                      
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         elu_7[0][0]                      
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
Total params: 252351

Instead of storing the training data in memory, I used the generator to generate training set and validation set. I did the following preprocessing and data augmentation in the generator:
1. cut the top 35 lines to eliminate trees & sky, and bottom 20 lines for the hood
2. resize the images to (66,200) to fit the Nvidia model
3. switch to YUV space to fit the Nvidia model
4. horizontal transit the images, and adjusting the steerings accordingly
5. randomly choosing left, center, or right camera, and adjusting the steerings accordingly
6. randomly fliping the image, and adjusting the steerings accordingly
7. in the training data generator, data with too small steerings are less likely to be chosen, otherwise the model would tend to make the car go straight

after hours of fine-tuning, the following parameters generated the best results:
batch_size = 256
pr_threshold = 0.75
number of epoches = 9
optimizer = adam with 0.0001 learning rate 
