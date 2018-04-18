
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import nltk
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('ggplot')
data = pd.read_csv('ner_11.csv', sep="|",quotechar='\n', lineterminator='\n', encoding='latin-1', names=['Word', 'Tag'])


# In[3]:


data.Word, data.Tag = data.Word.str.split(' ', 1).str 


# In[4]:


data = data.fillna(method="ffill")


# In[5]:


words = list(set(data["Word"].values))
n_words = len(words); 


# In[6]:


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


# In[7]:


data_1 = POS(data.Word)
data_1[data.Word == "/n"] = ''
data_1['Tag'] = data.Tag


# In[8]:


data = data_1


# In[9]:


words = list(set(data["Word"].values))


# In[10]:


n_words = len(words);


# In[11]:


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


# In[12]:


data_2 = Senttokener(data)


# In[13]:


data_2 = data_2[['Sentence #', 'Word', 'POS', 'Tag']]


# In[14]:


words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words);


# In[15]:


tags = list(set(data["Tag"].values))
n_tags = len(tags);


# In[16]:


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


# In[17]:


getter = Collector(data)


# In[18]:


sent = getter.get_next()


# In[19]:


sentences = getter.sentences


# In[20]:


max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[21]:


X = [[word2idx[w[0]] for w in s] for s in sentences]


# In[22]:


X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)


# In[23]:


y = [[tag2idx[w[2]] for w in s] for s in sentences]


# In[24]:


y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[25]:


y = [to_categorical(i, num_classes=n_tags) for i in y]


# In[26]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)


# In[27]:


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_len, mask_zero=True)(input) 
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  
model = TimeDistributed(Dense(50, activation="relu"))(model)  
crf = CRF(n_tags)
out = crf(model)


# In[28]:


model = Model(input, out)


# In[29]:


model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])


# In[30]:


model.summary()


# In[31]:


history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=20,
                    validation_split=0.1, verbose=1)


# In[32]:


hist = pd.DataFrame(history.history)


# In[33]:


plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["acc"], label='Training Accuracy')
plt.plot(hist["val_acc"], label='Test Accuracy')
plt.legend()
plt.title('Accuracy vs Epochs')
plt.show()

