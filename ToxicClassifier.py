
# coding: utf-8

# In[ ]:

import sklearn
import numpy as np
from numpy import random
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
import time
from nltk.corpus import stopwords;


# In[2]:

# In[ ]:

# training_data = 'TrainingData.xlsx'
# testing_data = 'TestingData.xlsx'
# train_xl = pd.ExcelFile(training_data)
# sheet_names = train_xl.sheet_names
# sheet_names[0]
# df = train_xl.parse('train')
# df.to_pickle('training_data.pkl')
# df.iloc[1:15]


# In[ ]:

# dates = pd.date_range('1/1/2000', periods=8)
# df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])
# df.iloc[1,1]


# In[4]:

# train_df = pd.read_pickle('training_data.pkl')
# #train_df.iloc[1:15]
# toxic_col = train_df['toxic']
# #toxic_col[1:15]
# toxic_comments = train_df.loc[toxic_col == 1]
# #toxic_comments[1:15]
# non_toxic_comments = train_df.loc[toxic_col == 0]
# #non_toxic_comments[1:15]
# toxic_comments.to_pickle('toxic_comments.pkl')
# non_toxic_comments.to_pickle('non_toxic_comments.pkl')

toxic_comments_f = open('toxic_comments.pkl','rb')
non_toxic_comments_f = open('non_toxic_comments.pkl','rb')

toxic_comments = pickle.load(toxic_comments_f)
non_toxic_comments = pickle.load(non_toxic_comments_f)

toxic_comments_f.close()
non_toxic_comments_f.close()

toxic_comments = list(toxic_comments['comment_text'])
non_toxic_comments = list(non_toxic_comments['comment_text'])


# In[10]:

tox_words = []
cln_words = []
documents = []
allowed_word_types = ["JJ","JJR","JJS","NN", "NNS","RB","RBR","VB","VBD", "VBG", "VBN", "VBP", "VBZ"]
stop_words = set(stopwords.words('english'))

i = 0
for p in toxic_comments:
    if type(p) is not str:
        continue
    if i > 10000:
        break
    documents.append((p, "tox"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    if (i % 1000 == 0):
        print(i / 1000)
    i = i + 1
    for w in pos:
        if w[1] in allowed_word_types and w[1] not in stop_words:
            tox_words.append(w[0].lower())

i = 0
for p in non_toxic_comments:
    if type(p) is not str:
        continue
    if i > 10000:
        break
    documents.append((p, "cln"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    if (i % 1000 == 0):
        print(i / 1000)
    i = i + 1
    for w in pos:
        if w[1] in allowed_word_types and w not in stop_words:
            cln_words.append(w[0].lower())





# In[17]:
print('length of tox comments:')
print(len(tox_words))

print('length of cln comments:')
print(len(cln_words))



# In[20]:

save_documents = open("labeled_data/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()
tox_words = nltk.FreqDist(tox_words)
cln_words = nltk.FreqDist(cln_words)


# word_features = list(all_words.keys())[:10000]
word_features = list(tox_words.keys())[:1000]
# print("Tox Words:")
# print(list(tox_words.keys())[:100])
# print("Cln Words:")
# print(list(cln_words.keys())[:100])
word_features = list(set(word_features + list(cln_words.keys())[:1000]))

print('length of word features:')
print(len(word_features))
print('tox word features:')
print(word_features[:100])
print('cln word_features')
print(word_features[1001:1101])

save_word_features = open("labeled_data/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



# In[ ]:




# In[21]:

def find_features(document):

    words = word_tokenize(document)
    words = set(words)
    features = {}
    for w in word_features:
        features[w] = (w.lower() in words)

    return features
print('complete documents')


# In[ ]:

# In[ ]:

#featuresets = [(find_features(comment), toxicity) for (comment, toxicity) in documents]
featuresets = []

i = 0
for tup in documents:
    if i % 1000 == 0:
        print(i)
    i = i + 1
    try:
        featuresets.append((find_features(tup[0]), tup[1]))
    except:
        print('excepted')
print('completed featuresets')


random.shuffle(featuresets)
print('length featuresets:')
print(len(featuresets))

save_featuresets = open('labeled_data/featuresets5k.pickle','wb')
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

testing_set = featuresets[13000:]
training_set = featuresets[:13000]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(100)

###############
save_classifier = open("classifiers/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


# In[ ]:

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

save_classifier = open("classifiers/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("classifiers/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("classifiers/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("classifiers/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

save_classifier = open("classifiers/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()


# In[ ]:




# In[ ]:




# In[ ]:



