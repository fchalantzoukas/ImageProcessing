import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from conf import *
from data import *

#Read MNIST data from idx files
(trainimages, trainlabels, testimages, testlabels) = importData()
classes=10

#Plot 1 image per class
figure, axis = plt.subplots(5, 2, constrained_layout = True)
figure.set_figwidth(4)
figure.set_figheight(10)
for i in range(classes):
    j=0
    while int(trainlabels[j])!=i:
        j+=1
    axis[i%5,i//5].imshow(trainimages[j], cmap=plt.cm.binary)
    axis[i%5,i//5].set_title("Number "+str(i))
plt.show()

#Resize and Normalize images
trainimages = trainimages.astype("float32")/255
testimages = testimages.astype("float32")/255
trainimages = np.expand_dims(trainimages, -1)
testimages = np.expand_dims(testimages, -1)

#Define classes and parameters
trainlabels = keras.utils.to_categorical(trainlabels, classes)
testlabelscat = keras.utils.to_categorical(testlabels, classes)
batch_size=50
epochs=200

#Neural Network Architecture
model = keras.Sequential(
    [
        keras.Input(shape=(28,28,1)),
        layers.Conv2D(6, kernel_size=(3, 3), strides=1, padding='same', activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(classes, activation="softmax"),
    ])
model.summary()

#Train Model and Plot Stats
model.compile(loss="mean_squared_error", optimizer="SGD", metrics=["accuracy"])
history = model.fit(trainimages, trainlabels, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(testimages, testlabelscat))
figure, axis = plt.subplots(2, 1)
x=np.linspace(1,epochs,epochs)
axis[0].plot(x,history.history['val_loss'], '-b')
axis[0].set_title("Loss")
axis[0].grid()
axis[1].plot(x,history.history['val_accuracy'], '-b')
axis[1].set_title("Accuracy")
axis[1].grid()
figure.tight_layout()
plt.show()

#Predict the labels of the test set and convert prediction matrices to single integers
predictions=model.predict(testimages).argmax(axis=1)

#Confusion Matrix
(diff, confusion) = confusionMatrix(classes, predictions, testlabels)
print('{:.2f}% accuracy'.format(100-100*diff/len(predictions)))
fig, ax = plt.subplots()
ax.matshow(confusion, cmap='YlGn')

ax = addText(classes, confusion, ax)
fig.set_figwidth(8)
fig.set_figheight(8)
plt.ylabel('Actual Number')
plt.xlabel('Predicted Number')
plt.title('Confusion Matrix')

plt.show()