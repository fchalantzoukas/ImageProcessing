import idx2numpy
from skimage.feature import hog

#Reads the idx-ubyte files and returns numpy arrays
def importData():
    file1 = 'data/train-images.idx3-ubyte'
    file2 = 'data/train-labels.idx1-ubyte'
    tfile1 = 'data/t10k-images.idx3-ubyte'
    tfile2 = 'data/t10k-labels.idx1-ubyte'
    trainimages = idx2numpy.convert_from_file(file1)
    trainlabels = idx2numpy.convert_from_file(file2)
    testimages = idx2numpy.convert_from_file(tfile1)
    testlabels = idx2numpy.convert_from_file(tfile2)
    return (trainimages, trainlabels, testimages, testlabels)

#Returns a list with the HOG Features per image
def extractHogFeatures(images, pixels_per_cell, cells_per_block):
    hogImages=[]
    for image in images:
        hogImage=hog(image, orientations=9, pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, channel_axis=-1)
        hogImages.append(hogImage)
    return hogImages