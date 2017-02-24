[image1]: ./simulator.jpeg "Project Picture"
[image2]: ./Nvidia-architecture.png "Nvidia Model"

# UDACITY Driving Behavior Cloning Project

## The Goal of This Project
The goal of this project to use convolutional neural networks to clone driving behavior using Keras. The model will output a steering angle to an autonomous vehicle in the simulator provided by Udacity.

## Summary
I used image data and steering angles collected in the simulator to train a neural network and then use this model to drive the car autonomously around the track. Check _model.py_ for the code.

![alt text][image1]

The project repo from Udacity can be found [here](https://github.com/udacity/CarND-Behavioral-Cloning-P3)

## The Model
Nvidia published an [end-to-end Convolutional Neural Networks](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) to process visual data. I applied the archetecture of the Nvidia network in this project. 

![alt text][image2]

Instead of storing the training data in memory, I used the generator to generate training set and validation set. I did the following preprocessing and data augmentation in the generator:

1. cut the top 35 lines to eliminate trees & sky, and bottom 20 lines for the hood
2. resize the images to (66,200) to fit the Nvidia model
3. switch to YUV space to fit the Nvidia model
4. horizontal transit the images, and adjusting the steerings accordingly
5. randomly choosing left, center, or right camera, and adjusting the steerings accordingly
6. randomly fliping the image, and adjusting the steerings accordingly
7. in the training data generator, data with too small steerings are less likely to be chosen, otherwise the model would tend to make the car go straight

After hours of fine-tuning, the following parameters generated the best results:
* batch_size = 256
* pr_threshold = 0.75
* number of epoches = 9
* optimizer = adam with 0.0001 learning rate 

I also added 2 dropout layers with 0.5 dropout rate in first 2 fully-connected layers to prevent overfitting.

## The Outcome
In the autonomous mode, the car is within the drivable portion of the track all the time. You can see the performance of the vehicle in _output_video.mp4_. 

