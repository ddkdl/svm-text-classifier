# Author Alvaro Esperanca

from SVM import SVM
from Validator import Validator
from PolynomialKernel import PolynomialKernel
import numpy as np
import sys

def main(args):
    if len(args) != 2:
        print "Insufficient arguments provided"
        print "Terminating..."
        return 

    trainFilename = args[0]
    testFilename = args[1]
    resutlsFilename = args[1].split('.')[0] + '_linear_results.txt'
    resultStatsFilename = args[1].split('.')[0] + '_linear_results_stats.txt'

    # Loading train and test data
    train_data = np.genfromtxt(trainFilename, dtype=int ,delimiter='\t')
    test_data = np.genfromtxt(testFilename, dtype=int, delimiter='\t')

    # print train_data
    # return

    trainEndIndex = len(train_data[0]) - 1
    testEndIndex = len(test_data[0]) - 1

    tempData = list()
    tempLabels = list()
    tempTest = list()
    tempTestLabels = list()

    for i in range(len(train_data)):
        tempData.append(train_data[i][0:trainEndIndex])
        if train_data[i][trainEndIndex] == 0:
            tempLabels.append(-1)
        else:
            tempLabels.append(1)

    for i in range(len(test_data)):
        tempTest.append(test_data[i][0:testEndIndex])
        if test_data[i][testEndIndex] == 0:
            tempTestLabels.append(-1)
        else:
            tempTestLabels.append(1)


    training_data = np.array(tempData)
    training_labels = np.array(tempLabels)

    testing_data = np.array(tempTest)
    validationLabels = np.array(tempTestLabels, 'd')


    clf = SVM(kernel=PolynomialKernel(p=trainEndIndex), C=1.0)
    val = Validator()

    clf.fit(training_data, training_labels)
    predictions = clf.predict(testing_data)

    val.validate(validationLabels, predictions)

    predFile = open(resutlsFilename, "w")
    statFile = open(resultStatsFilename, "w")

    predFile.write("Predicted\tActual\n")
    for i in range(len(predictions)):
        predFile.write("%d\t%d\n" % (predictions[i],validationLabels[i]) )
    
    statFile.write("%-20s %-5d\n" % ("True Positives:", val.truePositives()) )
    statFile.write("%-20s %-5d\n" % ("True Negatives:", val.trueNegatives()) )
    statFile.write("%-20s %-5d\n" % ("False Positives:", val.falsePositives()) )
    statFile.write("%-20s %-5d\n\n" % ("False Negatives:", val.falseNegatives()) )
    statFile.write("%-20s %-2.2f\n" % ("Accuracy:", val.accuracy()) )
    predFile.close()
    statFile.close()

if __name__ == "__main__":
    main(sys.argv[1:])