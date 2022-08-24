
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
import pickle

main = tkinter.Tk()
main.title("Decision Tree and Support Vector Machine for Anomaly Detection in Water Distribution Networks") #designing main screen
main.geometry("1300x1200")

global filename
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
global X
global Y
global classifier
global dataset


def upload(): 
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,encoding='iso-8859-1')
    text.insert(END,str(dataset.head()))

def preprocess():
    text.delete('1.0', END)
    global X
    global Y
    global dataset
    global X_train, X_test, y_train, y_test
    Y = []
    before_features = dataset.shape[1]
    dataset.drop(['STATION CODE','LOCATIONS','STATE'], axis = 1,inplace=True)
    dataset['Temp'] = pd.to_numeric(dataset['Temp'], errors='coerce')
    dataset['B.O.D. (mg/l)'] = pd.to_numeric(dataset['B.O.D. (mg/l)'], errors='coerce')
    dataset['NITRATE0'] = pd.to_numeric(dataset['NITRATE0'], errors='coerce')
    dataset['FECAL COLIFORM (MPN/100ml)'] = pd.to_numeric(dataset['FECAL COLIFORM (MPN/100ml)'], errors='coerce')
    dataset['TOTAL COLIFORM (MPN/100ml)Mean'] = pd.to_numeric(dataset['TOTAL COLIFORM (MPN/100ml)Mean'], errors='coerce')
    dataset.fillna(0, inplace = True)
    dataset = dataset.astype(float)
    X = dataset.values
    for i in range(len(X)):
        ecoli = X[i,6]
        if ecoli < 150:
            Y.append(0)
        else:
            Y.append(1)

    Y = np.asarray(Y)
    X = normalize(X) #dataset normalization
    text.insert(END,str(dataset.head()))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def runSVM():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    accuracy.clear()
    precision.clear()
    recall.clear()

    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    svm_acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    accuracy.append(svm_acc)
    precision.append(p)
    recall.append(r)
    text.insert(END,"SVM Prediction Accuracy : "+str(svm_acc)+"\n")
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall : "+str(r)+"\n")
    
def runKNN():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    accuracy.clear()
    precision.clear()
    recall.clear()

    cls=KNeighborsClassifier(n_neighbors = 3)
    cls.fit(X_train,y_train)
    predict = cls.predict(X_test)
    clf_knn_acc=cls.score(X_test,y_test)
    print('With KNN (K=3) accuracy is: ',cls.score(X_test,y_test))
    neighbor = np.arange(1,25)
    train_accuracy = []
    test_accuracy = []

    for i,k in enumerate(neighbor):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        p = precision_score(y_test, predict,average='macro') * 100
        r = recall_score(y_test, predict,average='macro') * 100
        knn_acc = accuracy_score(y_test,predict)*100
    accuracy.append(knn_acc)
    precision.append(p)
    recall.append(r)
    text.insert(END,"KNN Prediction Accuracy : "+str(knn_acc)+"\n")
    text.insert(END,"KNN Precision : "+str(p)+"\n")
    text.insert(END,"KNN Recall : "+str(r)+"\n")
    


def runDecisionTree():
    global X_train, X_test, y_train, y_test
    global classifier
    cls = DecisionTreeClassifier(random_state=42)
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    dt_acc = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    accuracy.append(dt_acc)
    precision.append(p)
    recall.append(r)
    text.insert(END,"Decision Tree Prediction Accuracy : "+str(dt_acc)+"\n")
    text.insert(END,"Decision Tree Precision : "+str(p)+"\n")
    text.insert(END,"Decision Tree Recall : "+str(r)+"\n")
    with open('model/model.txt', 'rb') as file:
        classifier = pickle.load(file)
    file.close()
    
    


def graph():
    df = pd.DataFrame([['Decision Tree','Precision',precision[1]],['Decision Tree','Recall',recall[1]],['Decision Tree','Accuracy',accuracy[1]],
                       ['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','Accuracy',accuracy[0]],
                       ['KNN','Precision',precision[2]],['KNN','Recall',recall[2]],['KNN','Accuracy',accuracy[2]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()   

def predict():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename,encoding='iso-8859-1')
    test = test.values
    #test = test.reshape((test.shape[0], test.shape[1], 1))
    y_pred = classifier.predict(test)
    print(y_pred)
    for i in range(len(test)):
        #y_pred = np.argmax(y_pred[i])
        predict = y_pred[i]  #np.argmax(y_pred[i])
        if predict == 0:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No Anomaly Detected in Water')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Anomaly Presence Detected in Water')+"\n\n")

font = ('times', 16, 'bold')
title = Label(main, text='Decision Tree and Support Vector Machine for Anomaly Detection in Water Distribution Networks')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Water Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess & Normalize Dataset", command=preprocess)
processButton.place(x=290,y=550)
processButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=570,y=550)
svmButton.config(font=font1) 

KNNButton = Button(main, text="Run KNN Algorithm", command=runKNN)
KNNButton.place(x=570,y=660)
KNNButton.config(font=font1) 


dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=50,y=600)
dtButton.config(font=font1)


graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=290,y=600)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Water Anomaly Detection", command=predict)
predictButton.place(x=570,y=600)
predictButton.config(font=font1) 

main.config(bg='sea green')
main.mainloop()
