#import matplotlib as mpl
import numpy as np
'''IMPORTANT NOTE: THE RESULTS OF THIS CODE WILL BE A LITTLE MESSY THAT IS WHY 
I UPLOADED THE MOST IMPORTNAT SCREENSHOTS IN THE REPORT'''

#1- we need to import the data from the file
def getDataFromFile():
    data = np.genfromtxt('dataset.csv', delimiter=',')
    return data

#this function splits the data into training,validation and 20% testing 
#we get the classes from the class function and add them to training,validation and testing arrays
#notice the equal proptions of each class
def splitData(data):
        np.random.shuffle(data)
        #print(data)
        class1,class2,class3,class4,class5 = getDataClasses(data)
        training = []
        testing = []
        validation = []
        #class 1 gets 10% validation (4 examples)
        for i in range(len(class1)):
            if i < len(class1)*70/100:
                training.append(class1[i])
            elif i < len(class1)*80/100:
                validation.append(class1[i])
            else:
                testing.append(class1[i])
        #5% for class 2 (4 examples)
        for i in range(len(class2)):
            if i < len(class2)*75/100:
                training.append(class2[i])
            elif i < len(class2)*80/100:
                validation.append(class2[i])
            else:
                testing.append(class2[i])
        #8% for class3 (4 examples)
        for i in range(len(class3)):
            if i < len(class3)*70/100:
                training.append(class3[i])
            elif i < len(class3)*80/100:
                validation.append(class3[i])
            else:
                testing.append(class3[i])
        #15% for class4
        for i in range(len(class4)):
            if i < len(class4)*65/100:
                training.append(class4[i])
            elif i < len(class4)*80/100:
                validation.append(class4[i])
            else:
                testing.append(class4[i])
        #20% for class5
        for i in range(len(class5)):
            if i < len(class5)*60/100:
                training.append(class5[i])
            elif i < len(class5)*80/100:
                validation.append(class5[i])
            else:
                testing.append(class5[i])
        training = np.array(training)
        validation = np.array(validation)
        testing = np.array(testing)
        '''
        print(len(training))
        print(len(validation))
        print(len(testing))
        print('Training')
        for i in training:
            print(i)
        print('Validation')
        for i in validation:
            print(i)    
        print('Testing')
        for i in testing:
            print(i) 
        '''
        return training,validation,testing

#we send this function data from getDataFromFile that returns an arrauy
#of arrays that has each example. then we compare the labels and add to
#a class array accordingly
def getDataClasses(data):
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    for example in data:
        if example[2] == 1:
            class1.append(example)
        elif example[2] == 2:
            class2.append(example)
        elif example[2] == 3:
            class3.append(example)
        elif example[2] == 4:
            class4.append(example)
        elif example[2] == 5:
            class5.append(example)
    class1 = np.array(class1)
    class2 = np.array(class2)
    class3 = np.array(class3)
    class4 = np.array(class4)
    class5 = np.array(class5)
    #print(len(class1)+len(class2)+len(class3)+len(class4)+len(class5))
    return class1,class2,class3,class4,class5

#this function takes the training,validation and testing data and makes 5 copies of it 
#each has OVA for a class (class 1 against the others and such)
#this data is what we are going to use in the perceptron
def getBinaryData(data):
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    for example in np.copy(data):
        if example[2] == 1:
            class1.append(example)
        if example[2] != 1:
            example[2] = 0
            class1.append(example)
    for example in np.copy(data):
        if example[2] != 2:
            example[2] = 0
            class2.append(example)
        if example[2] == 2:
            example[2] = 1
            class2.append(example)
    for example in np.copy(data):
        if example[2] != 3:
            example[2] = 0
            class3.append(example)
        if example[2] == 3:
            example[2] = 1
            class3.append(example)
    for example in np.copy(data):
        if example[2] != 4:
            example[2] = 0
            class4.append(example)
        if example[2] == 4:
            example[2] = 1
            class4.append(example)
    for example in np.copy(data):
        if example[2] != 5:
            example[2] = 0
            class5.append(example)
        if example[2] == 5:
            example[2] = 1
            class5.append(example)
    class1 = np.array(class1)
    class2 = np.array(class2)
    class3 = np.array(class3)
    class4 = np.array(class4)
    class5 = np.array(class5)
    return class1,class2,class3,class4,class5
    
#our main function. the testing and tunning happens here.
#this function calls the classify example and update weights. return an array of weights
#we choose class to do the perceptron on. N and epochs are hyper-parameters
#Tchosen is for the training data we got from getBinary.Vchosen is validation
def findOptimalHyperParameters(TchosenClass,VchosenClass,NoOfEpochs,learningRate):
    #np.random.shuffle(TchosenClass)
    weights = [0,-0.1,0.1]
    TtrueLabels = []
    VtrueLabels = []
    for example in TchosenClass:
        TtrueLabels.append(example[2])
    for example in VchosenClass:
        VtrueLabels.append(example[2])
    #print(TtrueLabels)
    #122 35
    print('Testing on Validation set:')
    print('learning rate: ',learningRate)
    print('Total Number of Epochs: ',NoOfEpochs)
    #train perceptron on training
    for i in range(NoOfEpochs):
        TpredictedLabels = []
        VpredictedLabels = []
        for example in TchosenClass:
            exampleInfo = example[:2]
            predictedLabel = classifyExample(weights,exampleInfo)
            #print('hi i am :',predictedLabel)
            TpredictedLabels.append(predictedLabel)
            if predictedLabel != example[2]:
                weights = updateWeights(learningRate,weights,example,predictedLabel)
                #print(weights)
        #after training and geeting the wieghts from 1 epoch. we try the accuracy on validation
        for example in VchosenClass:
            exampleInfo = example[:2]
            predictedLabel = classifyExample(weights,exampleInfo)
            VpredictedLabels.append(predictedLabel)
            if predictedLabel != example[2]:
                weights = updateWeights(learningRate,weights,example,predictedLabel)
        #print(VpredictedLabels)
        #print(len(VpredictedLabels))
        #calclate the accuracy after each epoch
        accuracy = calculateAccuracy(VtrueLabels,VpredictedLabels)
        print('epoch No: ',i+1)
        print('accuracy: ',accuracy)
    return None

#this function recieves weights and one example and returns if it belongs
#to the class or not (predicts its class)
def classifyExample(weights,testExample):
    bias = weights[0]
    weights = weights[1:]
    result = np.dot(weights,testExample)
    if result + bias > 0:
        return 1
    else:
        return 0
    
#this function gets called if we get a wrong prediction
#weights is an array having the bias at index 0
#n in the learning rate
def updateWeights(n,weights,example,predictedLabel):
    #print('hi im update')
    weights[0] = weights[0] + (n*(example[2] - predictedLabel)*1)
    weights[1] = weights[1] + (n*(example[2] - predictedLabel)*example[0])
    weights[2] = weights[2] + (n*(example[2] - predictedLabel)*example[1])
    updatedWeights = [weights[0],weights[1],weights[2]]
    updatedWeights = np.array(updatedWeights)
    return updatedWeights

#calculate recall for one class (testing data)
def calculateRecall(trueLabels,predictedLabels):
    #gets true positives
    acPositivies = 0
    for i in trueLabels:
        if i == 1:
            acPositivies += 1
    #gets predicted true positives 
    prTruePositivies = 0
    for i in range(len(predictedLabels)):
        if predictedLabels[i] == 1:
            if predictedLabels[i] == trueLabels[i]:
                prTruePositivies += 1
    return prTruePositivies/acPositivies

#calculate precision for one class (testing data)
def calculatePrecision(trueLabels,predictedLabels):
    #gets predicted positivies
    prPositivies = 0
    for i in predictedLabels:
        if i == 1:
            prPositivies += 1
    #gets predicted true positives 
    prTruePositivies = 0
    for i in range(len(predictedLabels)):
        if predictedLabels[i] == trueLabels[i]:
            prTruePositivies += 1
    if prPositivies == 0:
        return 0
    else:
        return prTruePositivies/prPositivies
    
#calculate F1 for one class (testing data)
def calculateF1(recall,precision):
    if precision == 0:
        return 0
    F1 = (2*precision*recall)/(precision+recall)
    return F1

#calculate accuracy for one class (testing data)
def calculateAccuracy(trueLabels,predictedLabels):
    count = 0
    for i in range(len(trueLabels)):
        if trueLabels[i] == predictedLabels[i]:
            count += 1
    acc=count/len(trueLabels)
    return acc*100

#oversmaple takes the binary data of a class and copies it randomly until
#we reach the size of the bigger class (note that ALL in OVA will always have more examples)
def overSample(classExamples):
    majority = []
    minority = []
    for example in np.copy(classExamples):
        if example[2] == 1:
            minority.append(example)
        if example[2] == 0:
            majority.append(example)
    i=0
    #print(len(minority))
    #print(len(majority))
    copy = np.copy(minority)
    Range = len(majority)-len(copy)
    while i < Range:
        minority.append(minority[i])
        i=i+1
    np.random.shuffle(minority)
    minority = np.array(minority)
    majority = np.array(majority)
    #print(minority)
    #print(len(minority))
    #print(len(majority))
    data = np.vstack((minority,majority))
    print(len(data))
    return data

#undersmaple takes the binary data of a class and removes examples from ALL
#we reach the size of the lesser class (note that ALL in OVA will always have more examples)
def underSample(classExamples):
    majority = []
    minority = []
    for example in np.copy(classExamples):
        if example[2] == 1:
            minority.append(example)
        if example[2] == 0:
            majority.append(example)
    i=0
    j=0
    copy = np.copy(majority)
    Range = len(copy)-len(minority)
    while i < Range:
        np.random.shuffle(majority)
        del majority[j]
        i=i+1
    np.array(minority)
    np.array(majority)
    #print(len(majority))
    #print(len(minority))
    #print(majority)
    data = np.vstack((minority,majority))
    print(len(data))
    return data

#macro accuracy for testing
def totalAcc(a1,a2,a3,a4,a5):
    avgAcc=(a1+a2+a3+a4+a5)/5
    return avgAcc

#macro Recall for testing
def macroRecall(r1,r2,r3,r4,r5):
    avgRecall=(r1+r2+r3+r4+r5)/5
    return avgRecall

#macro Precision for testing
def macroPrecision(p1,p2,p3,p4,p5):
    avgPrecision=(p1+p2+p3+p4+p5)/5
    return avgPrecision

#macro F1 for testing
def macroF1(macroRecall,macroPrecison):   
    m=calculateF1(macroRecall,macroPrecison)
    return m

#we combine training and validation and send them here to train the optimal hyper-parameters
def trainOptimalOnData(chosenClass,optimalLearningrate,optimalEpochs):
    trueLabels = []
    optimalWeights = [0,-0.1,0.1]
    for example in chosenClass:
        trueLabels.append(example[2])
    for i in range(optimalEpochs):
        predictedLabels = []
        for example in chosenClass:
            exampleInfo = example[:2]
            predictedLabel = classifyExample(optimalWeights,exampleInfo)
            predictedLabels.append(predictedLabel)
            if predictedLabel != example[2]:
                optimalWeights = updateWeights(optimalLearningrate,optimalWeights,example,predictedLabel)
    return optimalWeights

#the function that does the magic. after we got our optimal data. we test them here
def PerceptronOnTesting(chosenClass,optimalWeights,optimalLearningRate,optimalEpochs):
    predictedLabels = []
    trueLabels = []
    for example in chosenClass:
        trueLabels.append(example[2])
    for i in range(optimalEpochs):
        for example in chosenClass:
            exampleInfo = example[:2]
            predictedLabel = classifyExample(optimalWeights,exampleInfo)
            if i == 4:
                predictedLabels.append(predictedLabel)
    accuracy = calculateAccuracy(trueLabels,predictedLabels)
    print('epoch No: ',i+1)
    print('accuracy: ',accuracy)
    predictedLabels = np.array(predictedLabels)
    trueLabels = np.array(trueLabels)
    recall = calculateRecall(trueLabels,predictedLabels)
    precision = calculatePrecision(trueLabels,predictedLabels)
    F1 = calculateF1(recall,precision)
    print('Recall: ',recall)
    print('precision: ',precision)
    print('F1: ',F1)
    return accuracy,precision,recall,F1

#testing part
data = getDataFromFile()
tr,va,ts = splitData(data)
Trc1,Trc2,Trc3,Trc4,Trc5 = getBinaryData(tr)
Vc1,Vc2,Vc3,Vc4,Vc5 = getBinaryData(va)
Tsc1,Tsc1,Tsc1,Tsc1,Tsc1 = getBinaryData(ts)

#some cases that could be tested
findOptimalHyperParameters(Trc1,Vc1,5,0.1)
findOptimalHyperParameters(Trc2,Vc2,5,0.5)
findOptimalHyperParameters(Trc3,Vc3,5,0.1)
findOptimalHyperParameters(Trc4,Vc4,5,0.1)
findOptimalHyperParameters(Trc5,Vc5,5,0.2)

#test recall-precision-f1
true=[1,0,1,0,1]
predict=[1,0,0,1,1]
calculateAccuracy(true,predict)
calculatePrecision(true,predict)
calculateRecall(true,predict)

#class 1: optimalLearningRate = 0.1
#class 2: optimalLearningRate = 0.5
#class 3: optimalLearningRate = 0.1
#class 4: optimalLearningRate = 0.1
#class 5: optimalLearningRate = 0.2
#optimal epochs = 5

#merge training and validation
TVc1 = np.concatenate((Trc1,Vc1))
TVc2 = np.concatenate((Trc2,Vc2))
TVc3 = np.concatenate((Trc3,Vc3))
TVc4 = np.concatenate((Trc4,Vc4))
TVc5 = np.concatenate((Trc5,Vc5))

#train with optimal epochs and learningrate on the whole data
optimalWeights1 = trainOptimalOnData(TVc1,0.1,5)
optimalWeights2 = trainOptimalOnData(TVc2,0.5,5)
optimalWeights3 = trainOptimalOnData(TVc3,0.1,5)
optimalWeights4 = trainOptimalOnData(TVc4,0.1,5)
optimalWeights5 = trainOptimalOnData(TVc5,0.2,5)

#doing the testing and getting th values for macro functions
C1accuracy,C1precision,C1recall,C1F1 = PerceptronOnTesting(TVc1,optimalWeights1,0.1,5)
C2accuracy,C2precision,C2recall,C2F1 = PerceptronOnTesting(TVc2,optimalWeights2,0.5,5)
C3accuracy,C3precision,C3recall,C3F1 = PerceptronOnTesting(TVc3,optimalWeights3,0.1,5)
C4accuracy,C4precision,C4recall,C4F1 = PerceptronOnTesting(TVc4,optimalWeights4,0.1,5)
C5accuracy,C5precision,C5recall,C5F1 = PerceptronOnTesting(TVc5,optimalWeights5,0.2,5)

avgRecall = macroRecall(C1recall,C2recall,C3recall,C4recall,C5recall)
avgPrecision = macroPrecision(C1precision,C2precision,C3precision,C4precision,C5precision)
avgF1 = macroF1(avgRecall,avgPrecision)
avgAccuracy = totalAcc(C1accuracy,C2accuracy,C3accuracy,C4accuracy,C5accuracy)

print('average accuracy: ', avgAccuracy)
print('average recall: ', avgRecall)
print('average precision: ', avgPrecision)
print('average F1: ', avgF1)