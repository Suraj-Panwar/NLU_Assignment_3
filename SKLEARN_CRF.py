
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
import nltk
from sklearn_crfsuite import CRF
from sklearn.cross_validation import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
import eli5
data = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU Assignmnet 3\\ner_11.csv', sep="|",quotechar='\n', lineterminator='\n', encoding='latin-1', names=['Word', 'Tag'])


# In[103]:


data.Word, data.Tag = data.Word.str.split(' ', 1).str 


# In[104]:


data = data.fillna(method="ffill")


# In[105]:


words = list(set(data["Word"].values))
n_words = len(words);


# In[106]:


def POS(df):
    arr = np.array(df)
    ak = nltk.pos_tag(arr)
    ak_0 = []
    ak_1 = []
    for item in ak:
        ak_0.append(item[0])
        ak_1.append(item[1])
        
    df_1 = pd.DataFrame(df)
    df_1['POS'] = pd.Series(ak_1)
    
    return df_1


# In[115]:


class Collector(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[108]:


data_1 = POS(data.Word)
data_1[data.Word == "/n"] = ''
data_1['Tag'] = data.Tag


# In[109]:


data = data_1
words = list(set(data["Word"].values))
n_words = len(words);
data_2 = data_2[['Sentence #', 'Word', 'POS', 'Tag']]


# In[110]:


def Senttokener(df):
    sent = 1
    df['Sentence #'] = 0
    for index,i in enumerate(df.Word):
        if i=='':
            sent +=1
            df['Sentence #'][index] = str('Sentence:')+str(' ')+ str(sent)
        else:
            df['Sentence #'][index] = str('Sentence:')+ str(' ')+str(sent)
    return df


# In[111]:


data_2 = Senttokener(data)


# In[116]:


get_sent = Collector(data)


# In[117]:


sent = get_sent.get_next()


# In[120]:


print(sent)


# In[121]:


sentences = get_sent.sentences


# In[124]:


def feature(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# In[125]:


X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]


# In[126]:


crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)


# In[127]:


pred = cross_val_predict(estimator=crf, X=X, y=y, cv=10)


# In[128]:


report = flat_classification_report(y_pred=pred, y_true=y)
print(report)


# In[129]:


###########################################################################################


# In[130]:


#crf.fit(X, y)


# In[131]:


#eli5.show_weights(crf, top=30)


# In[132]:


crf = CRF(algorithm='lbfgs',
          c1=10,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)


# In[133]:


pred = cross_val_predict(estimator=crf, X=X, y=y, cv=10)


# In[134]:


report = flat_classification_report(y_pred=pred, y_true=y)
print(report)


# In[135]:


#crf.fit(X, y)


# In[136]:


#eli5.show_weights(crf, top=30)

