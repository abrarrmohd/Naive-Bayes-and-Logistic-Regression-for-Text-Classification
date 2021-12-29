import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def doc_load(file):
    new_list = []
    all_files=file
    for fle in os.listdir(all_files):
        # open the file and then call .read() to get the text
        with open(os.path.join(all_files, fle),"r",errors='ignore') as f:
            text = f.read()
            new_list.append(text)
    return new_list

def text_clean(new_list):
    import re
    for i in range(len(new_list)):
        new_list[i]=new_list[i].replace('Subject:','')
        new_list[i]=new_list[i].replace('\n',' ')
        new_list[i] = re.sub(r'[^\w\s]', '', new_list[i])
        new_list[i] = ''.join(i for i in new_list[i] if not i.isdigit())
        new_list[i]=" ".join(new_list[i].split())
    return new_list

def bag_of_words(new_list):
    bag=[]
    for i in range(len(new_list)):
        for j in new_list[i].split():
            if j not in bag:
                bag.append(j) 
    return bag

def dataset_create_bog(address,train,total_bag,bog):
    ham=doc_load(address+'\ham')
    ham=text_clean(ham)  
    spam=doc_load(address+'\spam')
    spam=text_clean(spam)
    total=ham+spam
    if train:
        total_bag=bag_of_words(total)
    df=pd.DataFrame(0, index=np.arange(len(total)), columns=total_bag)
    for i in range(len(total)):
        word_list=total[i].split()
        for j in range(len(word_list)):
            c=word_list.count(word_list[j])
            try:
                if bog:
                    df.loc[i][word_list[j]]=c
                else:
                    df.loc[i][word_list[j]]=1
            except:
                continue
    df.loc[:len(ham)-1,'Y_classification']=1
    df.loc[len(ham):,'Y_classification']=0
    df = df.sample(frac=1).reset_index(drop=True)
    m,n=df.shape
    return df.iloc[:,:n-1].to_numpy(),df.iloc[:,n-1].to_numpy(),total_bag,df

if __name__ == "__main__":
    train_path=sys.argv[1]
    test_path=sys.argv[2]
    #for bag of words model
    X_train,Y_train,total_bag,df1=dataset_create_bog(train_path,True,'',True)
    X_test,Y_test,total_bag,df2=dataset_create_bog(test_path,False,total_bag,True)
    params = {
        "max_iter" : [50, 100, 200, 350, 500],
        "loss" : ["hinge", "log"],
        "alpha" : [0.0001, 0.001, 0.01, 0.05, 0.1],
        "penalty" : ["l2", "l1"],
    }
    clf = SGDClassifier()
    grid = GridSearchCV(clf, param_grid=params)
    grid.fit(X_train, Y_train)
    print("Best Parameters for Bag of Words Model are-->",grid.best_params_)
    y_pred = grid.predict(X_test)
    print("Accuracy for Bag of Words Model= {:.4f}".format(accuracy_score(Y_test, y_pred)))
    print("F1 score for Bag of Words Model= {:.4f}".format(f1_score(Y_test, y_pred, average="binary")))
    print("Precision score for Bag of Words Model= {:.4f}".format(precision_score(Y_test, y_pred, average="binary")))
    print("Recall score for Bag of Words Model= {:.4f}".format(recall_score(Y_test, y_pred, average="binary")))

    #for bernoulli model
    X_train,Y_train,total_bag,df1=dataset_create_bog(train_path,True,'',False)
    X_test,Y_test,total_bag,df2=dataset_create_bog(test_path,False,total_bag,False)
    params = {
        "max_iter" : [50, 100, 200, 350, 500],
        "loss" : ["hinge", "log"],
        "alpha" : [0.0001, 0.001, 0.01, 0.05, 0.1],
        "penalty" : ["l2", "l1"],
    }
    clf = SGDClassifier()
    grid = GridSearchCV(clf, param_grid=params)
    grid.fit(X_train, Y_train)
    print("Best Parameters for Bernoulli Model are-->",grid.best_params_)
    y_pred = grid.predict(X_test)
    print('Accuracy for Bernoulli Model= {:.4f}'.format(accuracy_score(Y_test, y_pred)))
    print("F1 score for Bernoulli Model= {:.4f}".format(f1_score(Y_test, y_pred, average="binary")))
    print("Precision score for Bernoulli Model= {:.4f}".format(precision_score(Y_test, y_pred, average="binary")))
    print("Recall score for Bernoulli Model= {:.4f}".format(recall_score(Y_test, y_pred, average="binary")))
