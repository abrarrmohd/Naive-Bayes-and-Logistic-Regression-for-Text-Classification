import pandas as pd
import numpy as np
import os
import sys
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

def dataset_create(address,train,total_bag):
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
            try:
                df.loc[i][word_list[j]]=1
            except:
                continue
    df.loc[:len(ham)-1,'Y_classification']=1
    df.loc[len(ham):,'Y_classification']=0
    return df,total_bag,ham,spam

def train_NB(total_bag,ham,spam,df):
    prior_prob={}
    tct_ham={}
    tct_spam={}
    cond_prob_ham={}
    cond_prob_spam={}
    vocab=total_bag
    total=ham+spam
    N_total=len(total)
    tct_total_ham=0
    tct_total_spam=0
    prior_prob_ham=len(ham)/len(total)
    prior_prob_spam=len(spam)/len(total)
    for t in vocab:
        tct_ham[t]=df.loc[:len(ham)-1,t].sum()
        tct_spam[t]=df.loc[len(ham):,t].sum()
    for t in bag_of_words(ham):
        tct_total_ham=tct_total_ham+df.loc[:len(ham)-1,t].sum()
    for t in bag_of_words(spam):    
        tct_total_spam=tct_total_spam+df.loc[len(ham):,t].sum()
    for t in vocab:
        cond_prob_ham[t]=(tct_ham[t]+1)/(len(vocab)+tct_total_ham)
        cond_prob_spam[t]=(tct_spam[t]+1)/(len(vocab)+tct_total_spam) 
    return prior_prob_ham,prior_prob_spam,cond_prob_ham,cond_prob_spam

def apply_NB(prior_prob_ham,prior_prob_spam,cond_prob_ham,cond_prob_spam,vocab,d):
    temp_wordlist=[]
    words=d.split()
    for i in words:
        if i in vocab:
            temp_wordlist.append(i)
    score_ham= np.log(prior_prob_ham)
    score_spam= np.log(prior_prob_spam)
    for t in vocab:
        if t in temp_wordlist:
            score_ham=score_ham+ np.log(cond_prob_ham[t])
            score_spam=score_spam+ np.log(cond_prob_spam[t])
        else:
            score_ham=score_ham+ np.log(1-cond_prob_ham[t])
            score_spam=score_spam+ np.log(1-cond_prob_spam[t])
    if score_ham>score_spam:
        return 1
    else:
        return 0
if __name__ == "__main__":
    train_path=sys.argv[1]
    test_path=sys.argv[2]

    df_train,total_bag,train_ham,train_spam=dataset_create(train_path,True,'')
    prior_prob_ham,prior_prob_spam,cond_prob_ham,cond_prob_spam=train_NB(total_bag,train_ham,train_spam,df_train)
    df_test,total_bag,test_ham,test_spam=dataset_create(test_path,False,total_bag)

    test_pass=0
    test_fail=0
    y_pred=np.array([])
    for d in test_ham:
        if apply_NB(prior_prob_ham,prior_prob_spam,cond_prob_ham,cond_prob_spam,total_bag,d)==1:
            y_pred=np.append(y_pred,[1])
        else:
            y_pred=np.append(y_pred,[0])
    for d in test_spam:
        if apply_NB(prior_prob_ham,prior_prob_spam,cond_prob_ham,cond_prob_spam,total_bag,d)==0:
            y_pred=np.append(y_pred,[0])
        else:
            y_pred=np.append(y_pred,[1])

    passed=0
    failed=0
    y_test=df_test.loc[:,'Y_classification'].to_numpy()
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]:
            passed=passed+1
        else:
            failed=failed+1

    print("Bernoulli Model Training Dataset-----------------------------------------",df_train)
    print("Accuracy for Bernoulli Model using Naive Bayes= {:.4f}".format((passed)/(passed+failed)))
    print("F1 score for Bernoulli Model using Naive Bayes= {:.4f}".format(f1_score(y_test, y_pred, average="binary")))
    print("Precision score for Bernoulli Model using Naive Bayes= {:.4f}".format(precision_score(y_test, y_pred, average="binary")))
    print("Recall score for Bernoulli Model using Naive Bayes= {:.4f}".format(recall_score(y_test, y_pred, average="binary")))
