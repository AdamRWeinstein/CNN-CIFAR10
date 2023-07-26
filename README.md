# CIFAR-10 Image Classification with Convolutional Neural Networks

This repository contains an implementation of a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using Keras.

## Requirements

- Python 3.8+
- Keras
- Matplotlib

## Dataset

The CIFAR-10 dataset is used, which is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories. The dataset is automatically downloaded and loaded into the program using the Keras API.

## Implementation

The CNN model is built using Keras and consists of three sets of Convolutional, MaxPooling, and Dropout layers, followed by a Flatten layer and two Dense layers. This takes inspiration from the VGG model detailed in https://arxiv.org/abs/1409.1556. The Conv2D layers use the 'relu' activation function and the 'he_uniform' kernel initializer. Dropout layers are used to prevent overfitting. The final Dense layer has 10 units, one for each class in the CIFAR-10 dataset.

The model is compiled with the 'adam' optimizer and the SparseCategoricalCrossentropy loss function. It is trained for 15 epochs.

The code also includes a commented-out section for data augmentation, which can be used to artificially increase the size of the training set and improve the model's performance. I had difficulty getting this to work properly, so I plan to explore this further at a later date. I expect that the windows of the kernel were too small leading to issues with transformations outside of the pooled regions.

After training, the model's accuracy on the training set and validation set is plotted for each epoch. Finally, the model's accuracy on the test set is computed and printed.

## Usage

To run the script, simply use the following command:

```
python cnn_cifar10.py
```

The script will automatically download the CIFAR-10 dataset, train the CNN model, plot the training and validation accuracy, evaluate the model on the test set, and print the test accuracy.

## Results

By implementing the VGG architecture and incorporating Dropout layers, we saw improved performance of our model on unseen data. This signifies that our model became better at generalizing from the training data to new, unseen data. We nearly reached 80% accuracy on the validation set, and could improve further with more tweaks or more epochs.

Final Output:

Epoch 1/15
1563/1563 [==============================] - 133s 85ms/step - loss: 1.6174 - accuracy: 0.4055 - val_loss: 1.1938 - val_accuracy: 0.5720

Epoch 2/15
1563/1563 [==============================] - 133s 85ms/step - loss: 1.1213 - accuracy: 0.6042 - val_loss: 0.9173 - val_accuracy: 0.6727

Epoch 3/15
1563/1563 [==============================] - 130s 83ms/step - loss: 0.9286 - accuracy: 0.6729 - val_loss: 0.8546 - val_accuracy: 0.6994

Epoch 4/15
1563/1563 [==============================] - 121s 78ms/step - loss: 0.8222 - accuracy: 0.7142 - val_loss: 0.7346 - val_accuracy: 0.7473

Epoch 5/15
1563/1563 [==============================] - 111s 71ms/step - loss: 0.7452 - accuracy: 0.7370 - val_loss: 0.7808 - val_accuracy: 0.7316

Epoch 6/15
1563/1563 [==============================] - 113s 72ms/step - loss: 0.6985 - accuracy: 0.7570 - val_loss: 0.7107 - val_accuracy: 0.7621

Epoch 7/15
1563/1563 [==============================] - 108s 69ms/step - loss: 0.6558 - accuracy: 0.7713 - val_loss: 0.6704 - val_accuracy: 0.7750

Epoch 8/15
1563/1563 [==============================] - 107s 69ms/step - loss: 0.6258 - accuracy: 0.7844 - val_loss: 0.7099 - val_accuracy: 0.7664

Epoch 9/15
1563/1563 [==============================] - 108s 69ms/step - loss: 0.5941 - accuracy: 0.7949 - val_loss: 0.6446 - val_accuracy: 0.7807

Epoch 10/15
1563/1563 [==============================] - 107s 68ms/step - loss: 0.5731 - accuracy: 0.8006 - val_loss: 0.6332 - val_accuracy: 0.7834

Epoch 11/15
1563/1563 [==============================] - 107s 68ms/step - loss: 0.5558 - accuracy: 0.8064 - val_loss: 0.6505 - val_accuracy: 0.7829

Epoch 12/15
1563/1563 [==============================] - 107s 68ms/step - loss: 0.5342 - accuracy: 0.8114 - val_loss: 0.6726 - val_accuracy: 0.7871

Epoch 13/15
1563/1563 [==============================] - 107s 69ms/step - loss: 0.5168 - accuracy: 0.8202 - val_loss: 0.6308 - val_accuracy: 0.7898

Epoch 14/15
1563/1563 [==============================] - 107s 68ms/step - loss: 0.5089 - accuracy: 0.8222 - val_loss: 0.6415 - val_accuracy: 0.7927

Epoch 15/15
1563/1563 [==============================] - 107s 68ms/step - loss: 0.4924 - accuracy: 0.8283 - val_loss: 0.6545 - val_accuracy: 0.7937

313/313 - 4s - loss: 0.6545 - accuracy: 0.7937 - 4s/epoch - 14ms/step

Test accuracy: 0.7936999797821045
