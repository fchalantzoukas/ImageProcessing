import numpy as np

#Returns a confusion matrix and the number of negative predictions
def confusionMatrix(classes, predictions, testlabelslist):
    diff=0
    confusion=np.zeros((classes,classes))
    for i in range(len(predictions)):
        if predictions[i]!=testlabelslist[i]:
            diff+=1
        confusion[testlabelslist[i]][predictions[i]]+=1
    return (diff, confusion)

#Fills the initial colored confusion matrix with numbers
def addText(classes, confusion, ax):
    for i in range(classes):
        for j in range(classes):
            val = int(confusion[j, i])
            ax.text(i, j, str(val), va='center', ha='center')