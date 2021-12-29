import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
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

def dataset_create(address,train,total_bag,bog):
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
    return df,total_bag

def dataFrame_to_numpy(df,train):
    if train:
        train, valid = train_test_split(df, test_size=0.3)
        train_matrix=train.to_numpy()
        valid_matrix=valid.to_numpy()
        m,n=train_matrix.shape
        x_train=train_matrix[:,:n-1]
        y_train_label=train_matrix[:,n-1]
        y_train_label=np.expand_dims(y_train_label, axis = -1 )
        x_0=np.ones((x_train.shape[0],1))
        x_train=np.hstack((x_0,x_train))
        m,n=valid_matrix.shape
        x_valid=valid_matrix[:,:n-1]
        y_valid_label=valid_matrix[:,n-1]
        y_valid_label=np.expand_dims(y_valid_label, axis = -1 )
        xv_0=np.ones((x_valid.shape[0],1))
        x_valid=np.hstack((xv_0,x_valid))
        return x_train,y_train_label,x_valid,y_valid_label
    else:
        test_matrix=df.to_numpy()
        m,n=test_matrix.shape
        x_test=test_matrix[:,:n-1]
        y_test_label=test_matrix[:,n-1]
        y_test_label=np.expand_dims(y_test_label, axis = -1 )
        x_0=np.ones((x_test.shape[0],1))
        x_test=np.hstack((x_0,x_test))
        return x_test,y_test_label

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train(x_train,y_train_label,lambdaa):
    alpha=0.05
    m,n=x_train.shape
    W=np.zeros((1,n))
    for i in range(100):
        X_W=np.dot(x_train,W.transpose())
        pred_Y=sigmoid(X_W)
        cost= y_train_label - pred_Y
        part1=(alpha * np.dot(x_train.transpose(),cost))
        part2=(alpha * lambdaa * W)
        W=W+part1.transpose()-part2 
    return W

def accuracy_score(result,y_label):
    passed,failed=0,0
    for i in range(len(result)):
        if result[i] >=0.5:
            result[i]=1
        else:
            result[i]=0
    for i in range(len(result)):
        if result[i]==y_label[i]:
            passed=passed+1
        else:
            failed=failed +1 
    return (passed/(passed+failed)),result

if __name__ == "__main__":
    train_path=sys.argv[1]
    test_path=sys.argv[2]
    #for bag of words model
    #training
    accuracy_dict={}
    lamda_list=[0.001, 0.01, 0.05, 0.075,0.3, 0.45, 0.5, 0.75]
    df_train,total_bag=dataset_create(train_path,True,'',True)
    x_train,y_train_label,x_valid,y_valid_label=dataFrame_to_numpy(df_train,True)
    for i in lamda_list:
        weights=train(x_train,y_train_label,i)
        #test on validation set
        validation_result=(sigmoid(np.dot(x_valid,weights.transpose())))
        accuracy_dict[i],_=accuracy_score(validation_result,y_valid_label)
    final_lambda=list(accuracy_dict.keys())[list(accuracy_dict.values()).index(max(accuracy_dict.values()))]
    x_train_final,y_train_label_final=dataFrame_to_numpy(df_train,False)
    weights_final=train(x_train_final,y_train_label_final,final_lambda)
    #test on test set
    df_test,total_bag=dataset_create(test_path,False,total_bag,True)
    x_test,y_test_label=dataFrame_to_numpy(df_test,False)
    test_result=(sigmoid(np.dot(x_test,weights_final.transpose())))
    accuracy,test_result=accuracy_score(test_result,y_test_label)
    print("Accuracy for Bag of Words Model= {:.4f}".format(accuracy))
    print("F1 score for Bag of Words Model= {:.4f}".format(f1_score(test_result,y_test_label , average="binary")))
    print("Precision score for Bag of Words Model= {:.4f}".format(precision_score(test_result, y_test_label, average="binary")))
    print("Recall score for Bag of Words Model= {:.4f}".format(recall_score(test_result, y_test_label, average="binary")))
    print("Lambda selected for Bag of Words Model= ",final_lambda)

    print("Bag of Words Model training Dataset--------------------------------------------------------------------------------------")
    print(df_train)

    #for bernoulli model
    #training
    accuracy_dict={}
    lamda_list=[0.001, 0.01, 0.05, 0.075,0.3, 0.45, 0.5, 0.75]
    df_train,total_bag=dataset_create(train_path,True,'',False)
    x_train,y_train_label,x_valid,y_valid_label=dataFrame_to_numpy(df_train,True)
    for i in lamda_list:
        weights=train(x_train,y_train_label,i)
        #test on validation set
        validation_result=(sigmoid(np.dot(x_valid,weights.transpose())))
        accuracy_dict[i],_=accuracy_score(validation_result,y_valid_label)
    final_lambda=list(accuracy_dict.keys())[list(accuracy_dict.values()).index(max(accuracy_dict.values()))]
    x_train_final,y_train_label_final=dataFrame_to_numpy(df_train,False)
    weights_final=train(x_train_final,y_train_label_final,final_lambda)
    #test on test set
    df_test,total_bag=dataset_create(test_path,False,total_bag,False)
    x_test,y_test_label=dataFrame_to_numpy(df_test,False)
    test_result=(sigmoid(np.dot(x_test,weights_final.transpose())))
    accuracy,test_result=accuracy_score(test_result,y_test_label)
    print("Accuracy for Bernoulli Model= {:.4f}".format(accuracy))
    print("F1 score for Bernoulli Model= {:.4f}".format(f1_score(test_result,y_test_label , average="binary")))
    print("Precision score for Bernoulli Model= {:.4f}".format(precision_score(test_result, y_test_label, average="binary")))
    print("Recall score for Bernoulli Model= {:.4f}".format(recall_score(test_result, y_test_label, average="binary")))
    print("Lambda selected for Bernoulli Model= ",final_lambda)

    print("Bernoulli Model training Dataset--------------------------------------------------------------------------------------")
    print(df_train)
