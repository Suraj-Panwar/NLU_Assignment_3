import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, style
import numpy as np
from sklearn.cross_validation import train_test_split as split_1
import nltk
from nltk.corpus import wordnet




df = pd.read_csv('C:\\Users\\suraj\\Desktop\\NLU Assignmnet 3\\ner_11.csv', sep="|",quotechar='\n', lineterminator='\n', encoding='latin-1', names=['entity', 'label'])



df.entity, df.label = df.entity.str.split(' ', 1).str 





train_data, test_data, train_label, test_label = split_1(df.entity, df.label, test_size =0.3 , random_state =0)




test_data, dev_data, test_label, dev_label = split_1(df.entity, df.label, test_size =0.5, random_state =0)




def captilize(df):
    df_1 = pd.DataFrame(df)
    df_1['caps'] = np.where(df_1['entity'].str.contains('A|B|C|D|E|F|G|H|I|J|I|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z')==True, 'C', 'NC')
    return df_1





final_df = pd.DataFrame(captilize(train_data))




final_df['label'] = train_label




final_df[final_df.entity == "/n"] = ''




final_df.to_csv('Final_caps.txt', sep=' ', index=False, header=False, line_terminator='\n',encoding='latin-1', doublequote=False)




final_test = final_df.drop(['caps'], axis=1)




final_test.to_csv('Final.txt', sep=' ', index=False, header=False, line_terminator='\n',encoding='latin-1')




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




final_df_POS = POS(train_data)
final_df_POS['label'] = train_label
final_df_POS[final_df_POS.entity == "/n"] = ''




final_df_POS.to_csv('Final_POS.txt', sep=' ', index=False, header=False, line_terminator='\n',encoding='latin-1', doublequote=False)




final_df_POS_caps = captilize(final_df_POS)
final_df_POS_caps = final_df_POS_caps.drop(['label'], axis=1)
final_df_POS_caps['label'] = train_label
final_df_POS_caps[final_df_POS_caps.entity == ""] = ''




final_df_POS_caps.to_csv('Final_POS_caps.txt', sep=' ', index=False, header=False, line_terminator='\n',encoding='latin-1', doublequote=False)




def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done."),len(model),(" words loaded!")
    return model




model = loadGloveModel('C:\\Users\\suraj\\Desktop\\NLU Assignmnet 3\\glove.6B.50d.txt')




def gloved(df, model):
    df['Glove'] = 0
    for step,i in enumerate(df.entity):
        if i in model:
            df['Glove'][step] = model[i]
        else:
            df['Glove'][step] = ''
    return df        




final_glove = gloved(df, model)




final_glove = final_glove[['entity', 'Glove', 'label']]
final_glove = final_glove.drop(['label'], axis=1)
final_glove['label'] = train_label
final_glove[final_glove.entity == ""] = ''




final_glove.to_csv('Final_glove.txt', sep=' ', index=False, header=False, line_terminator='\n',encoding='latin-1', doublequote=False)

