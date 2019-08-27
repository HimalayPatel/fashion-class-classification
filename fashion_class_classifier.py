import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fashion_train_df = pd.read_csv('P39-Fashion-MNIST-Datasets\\fashion-mnist_train.csv',sep = ',')
fashion_test_df = pd.read_csv('P39-Fashion-MNIST-Datasets\\fashion-mnist_test.csv',sep = ',')

# print(fashion_train_df.head())
# print(fashion_train_df.tail())
# print(fashion_test_df.head())
# print(fashion_test_df.tail())

# print(fashion_train_df.shape)
# print(fashion_test_df.shape)

training=np.array(fashion_train_df, dtype='float32')
testing=np.array(fashion_test_df, dtype='float32')

import random
# i = random.randint(1,60000)
# plt.imshow(training[i,1:].reshape(28,28))
# label = training[i,0]
# print(label)
# plt.show()

# Define the dimensions of the plot grid
W_grid = 15
L_grid = 15

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot returns the figure object and axes object
# we can use the axes object to plot specific figures at various locations
fig, axes = plt.subplots(L_grid, W_grid, figsize = (20,20))
n_training = len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid): # create evenly spaces variables
    for j in np.arange(0, L_grid):
        # Select a random number
        index = np.random.randint(0, n_training)
        # read and display an image with the selected index
        axes[i,j].imshow(training[index,1:].reshape((28,28)))
        axes[i,j].set_title(training[index,0], fontsize = 8)
        axes[i,j].axis('off')
plt.subplots_adjust(hspace=0.8)
plt.show()

X_train=training[:, 1:]/255
y_train=training[:,0]

X_test=testing[:, 1:]/255
y_test=testing[:,0]

from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=12345)

X_train=X_train.reshape(X_train.shape[0],*(28,28,1))
X_test=X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate=X_validate.reshape(X_validate.shape[0],*(28,28,1))

# print(X_train.shape)
# print(X_test.shape)
# print(X_validate.shape)

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model=Sequential()
cnn_model.add(Conv2D(32,3,3,input_shape = (28,28,1),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim = 32,activation = 'relu'))
cnn_model.add(Dense(output_dim = 10,activation = 'sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
cnn_model.load_weights('model_weights.h5')

# train the model
# epochs=50
# cnn_model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, verbose=1, validation_data=(X_validate,y_validate))

# to save the weights and model architecture
# # Save the weights
# cnn_model.save_weights('model_weights.h5')
# # Save the model architecture
# with open('model_architecture.json', 'w') as f:
#     f.write(cnn_model.to_json())

# to reuse the weights and model architecture
# from keras.models import model_from_json
# # Model reconstruction from JSON file
# with open('model_architecture.json', 'r') as f:
#     cnn_model = model_from_json(f.read())
# # Load weights into the new model
# cnn_model.load_weights('model_weights.h5')

evaluation = cnn_model.evaluate(X_test, y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

predicted_classes = cnn_model.predict_classes(X_test)
L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12,12))
for i in np.arange(0, L):
    for j in np.arange(0, W):
        axes[i,j].imshow(X_test[i].reshape(28,28))
        axes[i,j].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
        axes[i,j].axis('off')
plt.subplots_adjust(wspace=0.5)
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)
plt.show()

from sklearn.metrics import classification_report
num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))

