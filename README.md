# Agata Gałus podstawy nauki o danych Rozpoznawanie obrazów Fashion-MNIST

## Introduction

Fashion-MNIST problem is one of many problems that can be solved using machine learning methods and optimalization.

Fashion-MNIST is a collection of data on photos of Zalando fashion articles - consist of a training set of 60,000 examples 
and a test set of 10,000 examples. 
Each example is a 28x28 grayscale image, associated with a label from 10 classes describing the category of the photo.

The goal of the exercise is to implement a model for the classification of thumbnails of photos of clothes
i.e. identifying a category the object belongs to based on the image. We can do it using many different methods, 
machine learning models. We can compare our results with the results of similar algorithms in the section
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/ (many different options)

The problem with description is at:
https://github.com/zalandoresearch/fashion-mnist
I downloaded Test and training data from https://github.com/zalandoresearch/fashion-mnist.

## Methods

### KNN (k-nearest-neighbours method)

The first method I used in my programs to solve the problem is k-nearest-neighbors (KNN) method.
It's a discriminatory model, it models conditional distribution.
 
The program that implements this approach is here [rozwiązanie KNN](fashion_mnist_KNN_ag.py)

When running the program, firstly calculating the Manhattan distance happens. It tests the distance 
of test data (whose labels we are to specify) from training data (whose labels do we know).
    
Then, when we know calculated distances, the labels are sorted. Then I calculate the probability that an image
belongs to each category (using sorted labels and having a number of neighbors)
    
Finally, I calculated classification error.

### MLP - Neural network models (Multi-layer Perceptron)

My second method is the neural network.
The program that implements this approach is here [rozwiązanie MLP](fashion_mnist_MLP_ag.py)

I used the TensorFlow library.
At the beggining values need to be prepare. 

I want to have values between 0 and 1 instead of a 1-255 RGB values.
As well I use the "One hot" method to change the labels from 1-10 into an arrays. This means label 3
changes into [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,]. 


##### Architecture:
I used the sequential interface from  Keras model, the input is fashion data (input layer)

There are 3 hidden layers ( 512, 256 and 128 neurons) They use ReLU activation function (this function returns max (0, x))
(if the sum of the weighted elements entering the neuron is greater than 0 then the activation function simply returns this value.) 
For negative values, the ReLU function returns 0). There are also Dropout layers. They prevent the neural network from overfitting. 

The last is output layer. It consists of 10 neurons, because we want to predict a probability for 10 classes of clothes.
The activation function here is softmax. The output will be an array consisting of 10 values for our observations adding up to 1.


Optimization is done by algorithm Adam algorythm. 
A description of the available methods is at: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

Fit function traina the model. Its parameters:
   * Training set.
   * The variable to be predicted.
   * The number of epochs used to train the model (one epoch is the passage of the entire set through the network).
   * Parameter that sets the information that will be displayed when training the network.
   * A parameter that tells you how many observations pass at once during one run before an update of parameters occurs.
   * Validation set.

EarlyStopping method is made not to "overtrain" the model.
It monitors the test set's loss at the end of each epoch. If the loss is not decreasing, then network training is stopped.
Parameters:
  * What data we would like to monitor 
  * The number of epochs (if there is no decrease through this number of epochs, the algorythm stops)
  * Parameter defining how to display the information.
