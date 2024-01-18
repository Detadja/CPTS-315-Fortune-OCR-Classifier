import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from statistics import mean

#Reading files into variables
stoplist = open('C:\\Users\\denis\\Desktop\\PA3\\fortune-cookie-data\\stoplist.txt').read().splitlines()
traindata = open('C:\\Users\\denis\\Desktop\\PA3\\fortune-cookie-data\\traindata.txt').read().splitlines()
trainlabels = open('C:\\Users\\denis\\Desktop\\PA3\\fortune-cookie-data\\trainlabels.txt').read().splitlines()
testdata = open('C:\\Users\\denis\\Desktop\\PA3\\fortune-cookie-data\\testdata.txt').read().splitlines()
testlabels = open('C:\\Users\\denis\\Desktop\\PA3\\fortune-cookie-data\\testlabels.txt').read().splitlines()
# print(stoplist)
# print(traindata)
# print(trainlabels)
# print(testdata)
# print(testlabels)

#Combined training and testing data
traintestdata = traindata + testdata
#print(traintestdata)

#Preprocesses the data
v = CountVectorizer(stop_words = stoplist, token_pattern = r"(?u)\b[a-zA-Z0-9_']{2,}\b", binary = True)
proc_data = v.fit_transform(traintestdata)
#print(processed_data)
proc_traindata = proc_data.toarray()[:len(traindata)]
proc_testdata = proc_data.toarray()[len(traindata):len(traindata) + len(testdata)]
# print(proc_data.shape[0])
# print(len(proc_traindata))
# print(len(proc_testdata))
#print(proc_traindata[0][813])

#Algorithm/Perceptron implementation and learning
mistakes = [0] * 20
weights = [0] * len(proc_traindata[0])
learn_rate = 1

for epoch in range(20): #Max iterations
    for i in range(len(proc_traindata)): #Each row
        a = 0
        for j in range(len(proc_traindata[i])): #Multiply each column/value in the row with its individual weights, then the summation
            a += proc_traindata[i][j] * weights[j]
        if a <= 0: #If a is less than or equal to 0; if theres a mistake, update weights and biases
            mistakes[epoch] += 1
            for k in range(len(proc_traindata[i])): #Update weights corresponding to each value
                weights[k] += learn_rate * int(trainlabels[i]) * proc_traindata[i][k]

print(mistakes, '\n')
# print(weights, '\n')
# print(bias)

#Test data testing
predicted = [0] * len(proc_testdata)
for i in range(len(proc_testdata)): #Each row
    a = 0
    for j in range(len(proc_testdata[i])): #Multiply each column/value in the row with its individual weights, then the summation
        a += proc_testdata[i][j] * weights[j]
    if a <= 0: #If a is less than or equal to 0; if theres a mistake, predicted = 0
        predicted[i] = 0
    else: #If a is greater than 0; if its correct, predicted = 1
        predicted[i] = 1

# print(predicted, '\n')
# print(testlabels)
# print(len(predicted), '\n')
# print(len(testlabels))

#Calculating accuracy
str_pred = [str(x) for x in predicted]
accuracy = accuracy_score(testlabels, str_pred)
print('Accuracy: %.2f' %accuracy)

#-----------------------------------------------------------------------------------------------------------------

#Reading files into dataframe variables and splitting them
trainfile = pd.read_csv('C:\\Users\\denis\\Desktop\\PA3\\OCR-data\\ocr_train.txt', header = None)
testfile = pd.read_csv('C:\\Users\\denis\\Desktop\\PA3\\OCR-data\\ocr_test.txt', header = None)
ocr_train = trainfile[0].str.split('\t', expand = True)
ocr_train[1] = ocr_train[1].map(lambda x: x.lstrip('im'))
ocr_train = ocr_train.drop(columns = [0, 3])

ocr_test = testfile[0].str.split('\t', expand = True)
ocr_test[1] = ocr_test[1].map(lambda x: x.lstrip('im'))
ocr_test = ocr_test.drop(columns = [0, 3])

# print(ocr_train)
# print(ocr_test)

#Separating class values and labels from dataframe
ocr_trainlabel = ocr_train[2].values.tolist()
ocr_testlabel = ocr_test[2].values.tolist()

ocr_trainvalues = ocr_train[1].values.tolist()
ocr_testvalues = ocr_test[1].values.tolist()

#Separates the binary digits of each image to a list
ocr_traindigits = []
for i in ocr_trainvalues: 
    ocr_traindigits.append([*i])
ocr_testdigits = []
for i in ocr_testvalues:
    ocr_testdigits.append([*i])

# print(ocr_trainlabel)
# print(ocr_testlabel)
# print(len(ocr_trainvalues))
# print(len(ocr_testvalues))
# print(ocr_traindigits)
# print(ocr_testdigits)

#Each binary digit is turned into a column value in a dataframe
ocr_traindf = pd.DataFrame(ocr_traindigits)
ocr_testdf = pd.DataFrame(ocr_testdigits)
# print(ocr_traindf)
# print(ocr_testdf)

#Training the perceptron
eta = 0.05
ocr_accuracy = []
for i in range(20):
    ppn = Perceptron(max_iter = 20, eta0 = eta, random_state = 0, verbose = 1, shuffle = True)
    ppn.fit(ocr_traindf, ocr_trainlabel)
    ocr_predicted = ppn.predict(ocr_testdf)

    ocr_strpred = [str(x) for x in ocr_predicted]
    ocr_accuracy.append(accuracy_score(ocr_testlabel, ocr_strpred))
    eta += 0.05

ocr_accuracy = ['%.2f' % elem for elem in ocr_accuracy]
ocr_accuracy = [float(i) for i in ocr_accuracy]
avg_accuracy = mean(ocr_accuracy)
print('Accuracy:', ocr_accuracy)
print('Average Accuracy:', avg_accuracy)

#-----------------------------------------------------------------------------------------------------------------

#output
with open("C:\\Users\\denis\\Desktop\\PA3\\output.txt", "w") as ofile:
    count = 1
    for i in mistakes:
        ofile.write('iteration-' + str(count) + ' ' + str(i) + '\n')
        count += 1
    ofile.write('\n')
    count = 1
    for i in ocr_accuracy:
        ofile.write('iteration-' + str(count) + ' ' + str(i) + '\n')
        count += 1
