#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/bin/python

def train_classifier(X, y):
    from sklearn.linear_model import LogisticRegression
    cls=LogisticRegression(random_state=0, max_iter=10000)
    cls.fit(X, y)
    return cls

def evaluate(X, yt, cls, name='data'):
    from sklearn import metrics
    yp = cls.predict(X)
    acc = metrics.accuracy_score(yt, yp)
    print("  Accuracy on %s  is: %s" % (name, acc))


# In[4]:


def read_files(tarfname):
   
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    """" REPLACE COUNTS BY TFIDF OF WORDS"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.tfidf_vect = TfidfVectorizer(max_df=0.8,sublinear_tf=True, use_idf=True)
    sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    
    
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()

def read_unlabeled(tarfname, sentiment):
    
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.tfidf_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels



if __name__ == "__main__":
    print("Reading data")
    tarfname = "/Users/ashlesha_vaidya/Downloads/A2_256_sp19/data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    
    cls = train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
    print("\nReading unlabeled data")
    unlabeled = read_unlabeled(tarfname, sentiment)
    #write_pred_kaggle_file(unlabeled, cls, "/Users/ashlesha_vaidya/Desktop/cse 256/A2_256_sp19/data/sentimentpred.csv", sentiment)


# In[5]:


"""" HYPERPARAMTER TUNING"""

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
C=[0.01,0.1,1.0,10.0,100.0,200.0,1000.0,10000.0]
penalty=['l1','l2']
max_acc=0
final_penalty='l1'
final_C=0.1
for i in C:
    for p in penalty:
        logistic=LogisticRegression(penalty=p,C=i,random_state=0, max_iter=10000)
        logistic.fit(sentiment.trainX,sentiment.trainy)
        yp = logistic.predict(sentiment.devX)
        acc = metrics.accuracy_score(sentiment.devy, yp)
        print("  Accuracy on penalty= %s C= %s is: %s" % (p,i, acc))
        if acc> max_acc:
            max_acc=acc
            final_penalty=p
            final_C=i
            
            
#FINAL MODEL PERFORMACE ON TRAIN AND DEV SET
logistic=LogisticRegression(penalty='l2',C=10.0,random_state=0, max_iter=10000)
logistic.fit(sentiment.trainX,sentiment.trainy)
yp = logistic.predict(sentiment.trainX)
acc = metrics.accuracy_score(sentiment.trainy, yp)
print("  FINAL Accuracy on TRAIN SET penalty= l2 C= 10 is: %s" % ( acc))
yp = logistic.predict(sentiment.devX)
acc = metrics.accuracy_score(sentiment.devy, yp)
print("  FINAL Accuracy on DEV SET penalty= l2 C= 10 is: %s" % ( acc))


# In[157]:


#PLOTING THE PERFORMANCE OF THE MODEL AS A VARIATION OF HYPER-PARAMETERS

import seaborn as sns
sns.set(font_scale=2, style="white")
data=pd.DataFrame(columns={'C','acc','penalty'})
sns.set(rc={'figure.figsize':(11.7,8.27)})
data['C']=[0.01,0.1,1.0,10.0,100.0,200.0,1000.0,10000.0,0.01,0.1,1.0,10.0,100.0,200.0,1000.0,10000.0]
data['acc']=[50.0,56.77,75.99,76.63,74.01,74.67,75.32,75.32,74.23,75.32,79.25,81.00,78.38,78.16,77.29,76.63]
#data['acc2']=[74.23,75.32,79.25,81.00,78.38,78.16,77.29,76.63]
data['penalty']=['l1','l1','l1','l1','l1','l1','l1','l1','l2','l2','l2','l2','l2','l2','l2','l2']
sns.barplot(x = "C", y = "acc", hue='penalty',data = data);


# In[180]:


##FEATURE NAMES AND WEIGHTS
feats=sentiment.tfidf_vect.get_feature_names()
coefs=logistic.coef_[0]
ind=feats.index('cute')
print coefs[ind]

#MOST IMPORTANT FEATURES FOR THE NEGATIVE LABEL
pairs=zip(coefs,feats)
print sorted(pairs, reverse=True)[-3:]


# In[183]:


#MOST IMPORTTANT FEATURES FOR THE POSITIVE LABEL
pairs=zip(coefs,feats)
print sorted(pairs, reverse=True)[:3]


# In[6]:


from lime import lime_text


# In[7]:


from sklearn.pipeline import make_pipeline
c = make_pipeline(sentiment.tfidf_vect, logistic)


# In[18]:


#TRYING OUT DIFFERENT DEV DATA USING LIME
print (sentiment.dev_data[2])
print(c.predict_proba([sentiment.dev_data[2]]))


# In[22]:


from lime.lime_text import LimeTextExplainer
class_names=['NEGATIVE','POSITIVE']
explainer = LimeTextExplainer(class_names=class_names)


# In[59]:


idx = 50
#from pathlib 
exp = explainer.explain_instance(sentiment.dev_data[idx], c.predict_proba, num_features=14)
print (sentiment.dev_data[idx])
print('Document id: %d' % idx)
print('Probability =(POSITIVE)', c.predict_proba([sentiment.dev_data[idx]])[0,1])
print('True class: %s' % sentiment.dev_labels[idx])
print exp.as_list()
output="/Users/ashlesha_vaidya/Desktop/cse 256/output.html"
exp.save_to_file(output)



# In[ ]:





# In[ ]:




