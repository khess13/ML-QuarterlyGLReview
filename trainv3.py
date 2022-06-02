from sklearn.feature_extraction.text import TfidfVectorizer #CountVectorizer, TfidfTransformer,
from sklearn.model_selection import train_test_split#, cross_validate
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score

import pandas as pd
import numpy as np
import pickle
import os

#from nltk.corpus import stopwords
#stop = stopwords.words('english')
extra = ['inc', 'lp', '&', 'llc', 'sc', ',', '-']
stop = extra #+stop

'''targets'''
root = os.getcwd()
#fileloc = 'H:\\01 PMO Projects\\IT-RE GL Review\\text_class\\TrainingDataFY19PP12.csv' #TrainingDataFY18PP12.csv -- sw
#pickleloc = 'H:\\01 PMO Projects\\IT-RE GL Review\\text_class\\pickleJar\\'
#Training data -- Preprocess in personal folder
fileloc = root + '\\TrainingDatav2.csv'#'\\TrainingData.csv'#'\\PreprocessTrainingFY21Q2.csv'
pickleloc = root +'\\pickleJar\\'

no = int(input('0 for IT, 1 for RE'))
target = ['IT?', 'RE?']
#seedno = np.random.randint(0,9)
seedno = 309
#read file, dtype specs for faster read

acct_trans = pd.read_csv(fileloc,
                    header = 0,
                    usecols = ['Ven_LD_Header',
                               'VenTxt',
                               'Long Description',
                               target[no]],
                    dtype = {'Ven_LD_Header': str,
                             'VenTxt': str,
                             'Long Description': str,
                             target[no]: str},
                    encoding='iso-8859-1') #because fu excel

#to fix 'Integer value of NA in column x'
acct_trans.dropna(how = 'all', inplace = True) #acct_trans =
acct_trans[target[no]] = acct_trans[target[no]].apply(lambda x: int(x))
acct_trans = acct_trans.replace(np.nan, '', regex = True)

# large C value will choose smaller-margin hyperplane ---> strives to label data more finely
# small C value will choose larger-margin hyperplane ---> strives to label data more broadly
text_clf = Pipeline([
    ('vect', TfidfVectorizer(ngram_range = (1,3), stop_words = stop)), #, stop_words = stop)), #(1,2)
    ('clf', LinearSVC(C=0.5)) #0.3
])

#train for task without GL acct
#acct_trans['VenLD'] = acct_trans['VenTxt'] + ' ' + acct_trans['Long Description']

#create train/set sets
train_task = ['VenTxt','Ven_LD_Header'] #maybe add LD + GL text #'Long Description',
#train_task = ['VenLD']

for task in train_task:
    print(f'Training for {task}')
    d_train, d_test, l_train, l_test = train_test_split(acct_trans[task], #data
                                                        #acct_trans['VenTxt'],
                                                        acct_trans[target[no]], #labels
                                                        random_state = seedno)#,  #seed
                                                        #test_size = 0.3)

    #train
    features = text_clf.fit(d_train, l_train)
    #predict
    predicted = text_clf.predict(d_test)

    #how often the model is right against labels
    accuracy = np.round(accuracy_score(l_test, predicted) * 100, decimals = 2)
    #quality of labels
    recall = np.round(recall_score(l_test, predicted) * 100, decimals = 2)
    #data being trained vs total data set
    pct_target = np.round(np.mean(acct_trans[target[no]]) * 100, decimals = 2)
    print('% of data related to',target[no],': ', pct_target,'%')
    #print('Accuracy ',np.round(np.mean(predicted == l_test) * 100, decimals = 2),'%')
    print(f'Accuracy {accuracy}%')
    print(f'Precision of guess {recall}%')
    #cross validation
    #xval = cross_validate(text_clf, d_train, l_train, cv = 5)
    #print(f'Crossvalidation {xval}')

    '''Save model'''
    #pickle saves python objects, wb = write in bytes for >py2
    #to save
    save_classifier = open(pickleloc+task+target[no][:2]+'.pickle', 'wb') #fav_class
    pickle.dump(text_clf, save_classifier)
    save_classifier.close()
