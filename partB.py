import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import svm, metrics
from conf import *
from data import *

#Read MNIST data from idx files
(trainimages, trainlabels, testimages, testlabels) = importData()

#Resize and Normalize images
trainimages = trainimages.astype("float32")/255
testimages = testimages.astype("float32")/255
trainimages = np.expand_dims(trainimages, -1)
testimages = np.expand_dims(testimages, -1)

#Parameters
classes=10
pixels_per_cell=(2, 2)
cells_per_block=(2, 2)

#HOG extraction
start_time = time.time()
hogTrainImages=extractHogFeatures(trainimages, pixels_per_cell, cells_per_block)
hogTestImages=extractHogFeatures(testimages, pixels_per_cell, cells_per_block)

#Classification
classifier = svm.SVC(kernel='linear')
classifier.fit(hogTrainImages, trainlabels)
preds=classifier.predict(hogTestImages)
print('{:.2f} seconds'.format(time.time()-start_time))

#Confusion Matrix
(diff, confusion) = confusionMatrix(classes, preds, testlabels)
print('{:.2f}% accuracy'.format(100-100*diff/len(preds)))
fig, ax = plt.subplots()
ax.matshow(confusion, cmap='YlGn')

ax = addText(classes, confusion, ax)
fig.set_figwidth(8)
fig.set_figheight(8)
plt.ylabel('Actual Number')
plt.xlabel('Predicted Number')
plt.title('Confusion Matrix')

plt.show()