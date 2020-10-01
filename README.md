# Handwritten-digit-images-classification

**Dataset** : [THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)

**Implementation summary** :  
- Wrote a Flask application with kNN algorithm in the backend, for classifying the handwritten digit test images in PNG format, that are uploaded from the user interface.

- Wrote modules in Numpy, for forward/back prop, random/He initialization, mini-batch GD/Momentum/Adam optimizers, L2 regularization, dropout, and softmax output layer, to setup a neural network for the multinomial image classification task. When trained using Adam optimizer with dropout, the model obtained training accuracy of roughly 99.96% and validation accuracy of 94.59%.

- Developed a CNN model using Tensorflow for the classification task. The model achieved prediction accuracies of 96.62% and 96.34% on the training and validation sets respectively.
