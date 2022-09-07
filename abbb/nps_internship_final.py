import pandas as pd
import nltk
nltk.download('stopwords')

# Selecting a subset of data to be faster in demonstration
train_df = pd.read_csv('Train.csv').head(4000)
valid_df = pd.read_csv('Valid.csv').head(500)
test_df = pd.read_csv('Test.csv').head(500)
print('Train: '+ str(len(train_df)))
print('Valid: '+ str(len(valid_df)))
print('Test: '+ str(len(test_df)))
train_df.head(10)
# Turnig all text to lowercase
train_df['text'] = train_df['text'].str.lower()
valid_df['text'] = valid_df['text'].str.lower()
test_df['text'] = test_df['text'].str.lower()
train_df.head()
import string

exclude = set(string.punctuation)

def remove_punctuation(x):
    try:
        x = ''.join(ch for ch in x if ch not in exclude)
    except:
        pass
    return x

train_df['text'] = train_df['text'].apply(remove_punctuation)
valid_df['text'] = valid_df['text'].apply(remove_punctuation)
test_df['text'] = test_df['text'].apply(remove_punctuation)
train_df.head()
from nltk.corpus import stopwords

stop = stopwords.words('english')

train_df['text'] = train_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
valid_df['text'] = valid_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
test_df['text'] = test_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
train_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors for every sentence
vectorizer = TfidfVectorizer(#min_df = 5,
                             #max_df = 0.8,
                             max_features = 20000,
                             sublinear_tf = True,
                             use_idf = True)#, stop_words='english')#vocabulary = list(embeddings_index.keys()

train_vectors = vectorizer.fit_transform(train_df['text'])
valid_vectors = vectorizer.transform(valid_df['text'])
test_vectors = vectorizer.transform(test_df['text'])
from sklearn import svm
# SVM
classifier_linear = svm.SVC(kernel='linear')
#Train
classifier_linear.fit(train_vectors, train_df['label'])
from sklearn.metrics import classification_report

predictions = classifier_linear.predict(test_vectors)
# results
report = classification_report(test_df['label'], predictions)
print(report)
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Defining the NN model
model = Sequential()
model.add(Dense(20, input_shape=(train_vectors.shape[1],), activation='relu'))
model.add(Dropout(0.3))
#model.add(Dense(5, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model.h5',
        monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,verbose=1)
]

history = model.fit(
    train_vectors.toarray(), train_df['label'],
    epochs=20,
    verbose=1,
    callbacks = callbacks_list,
    validation_data=(valid_vectors.toarray(), valid_df['label']))

valid_vectors.toarray()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Test.csv')

dataset2 = pd.read_csv('Students Satisfaction Survey for NAAC_final.csv')
n = dataset2.shape[0]
dataset2.iloc[-2,-1] = "very nice"
x = dataset2.iloc[:,-1]
ab_vectors = vectorizer.transform(x)

n

y_xx = model.predict(ab_vectors.toarray())
y_xx

dataset88 = pd.DataFrame(dataset2.iloc[:,-1])
dataset88.columns = ['Suggestions']
dataset88.insert(1,"Sentiment(Positive/Negative)",y_xx)

dataset88

dataset3 = dataset2
df = pd.concat([dataset['text'],dataset['label']],axis=1)
df

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

wordnet = WordNetLemmatizer()
corpus = []
for i in range(0, 5000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
  review = review.lower()
  review = review.split()
  #ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [wordnet.lemmatize(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus)

"""## Creating the Bag of Words model"""

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
pos=0
neg=0
for i in range(0,5000):
  if(y[i]==0):
    pos=pos+1
  else:
    neg=neg+1
print(pos)
print(neg)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

"""## Training the Naive Bayes model on the Training set"""

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""# Training on Simple Logistic Regression




"""

from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred2)
print(cm)
accuracy_score(y_test, y_pred2)

"""# Training on SVM

"""

from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, y_train)
y_pred3 = classifier3.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred3)
print(cm)
accuracy_score(y_test, y_pred3)

"""# Training On Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
classifier4 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier4.fit(X_train, y_train)
y_pred4 = classifier4.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred4)

from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier5.fit(X_train, y_train)
y_pred5 = classifier5.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred5)
print(cm)
accuracy_score(y_test, y_pred5)

"""# Training on Multinomial Nb"""

from sklearn.naive_bayes import MultinomialNB
classifier6 = MultinomialNB()
classifier6.fit(X_train, y_train)
y_pred6 = classifier6.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred6)
print(cm)
accuracy_score(y_test, y_pred6)

from sklearn.ensemble import RandomForestClassifier
classifier7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier7.fit(X_train, y_train)
y_pred7 = classifier7.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred7)
print(cm)
accuracy_score(y_test, y_pred7)

"""## SVM has highest accuracy score"""

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred3))

dataset['corpus'] = corpus
dataset

dataset7 = pd.DataFrame(dataset2.iloc[:,-1])
dataset7.columns = ['Suggestions']

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus2 = []
for i in range(0, len(dataset7)):
  sugg = re.sub("[^a-zA-Z]", " ", dataset7['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  ps2 = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [ps2.stem(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus2.append(sugg)

Xncs = cv.transform(corpus2).toarray()

ypred_ncs = classifier3.predict(Xncs)

len(dataset7)

ypred_ncs
dataset7.insert(1,"Sentiment(Positive/Negative)",ypred_ncs)

for i in range(0,n):
  if dataset7['Sentiment(Positive/Negative)'][i] == 0:
    dataset7['Sentiment(Positive/Negative)'][i] = 'Negative'
  if dataset7['Sentiment(Positive/Negative)'][i] == 1:
    dataset7['Sentiment(Positive/Negative)'][i] = 'Positive'

dataset7

positive = 0
negative = 0
for i in range(0,n):
  if dataset7['Sentiment(Positive/Negative)'][i] == 'Negative':
    negative = negative + 1
  if dataset7['Sentiment(Positive/Negative)'][i] == 'Positive':
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/n)
print("Percentage negative responses/suggestions = ",negative*100/n)

dataset_cmpn = dataset2.loc[(dataset2['Program Name']=='CMPN')]
n_cmpn = dataset_cmpn.shape[0]
dataset_inft = dataset2.loc[(dataset2['Program Name']=='INFT')]
n_inft = dataset_inft.shape[0]
dataset_etrx = dataset2.loc[(dataset2['Program Name']=='ETRX')]
n_etrx = dataset_etrx.shape[0]
dataset_extc = dataset2.loc[(dataset2['Program Name']=='EXTC')]
n_extc = dataset_extc.shape[0]
dataset_biom = dataset2.loc[(dataset2['Program Name']=='BIOM')]
n_biom = dataset_biom.shape[0]
dataset_mms = dataset2.loc[(dataset2['Program Name']=='MMS')]
n_mms = dataset_mms.shape[0]

for i in range(0,n_cmpn):
  if dataset_cmpn.iloc[i,18] == '85 to 100%':
    dataset_cmpn.iloc[i,18] = 5
  if dataset_cmpn.iloc[i,18] == '70 to 84%':
    dataset_cmpn.iloc[i,18] = 4
  if dataset_cmpn.iloc[i,18] == '55 to 69%':
    dataset_cmpn.iloc[i,18] = 3
  if dataset_cmpn.iloc[i,18] == '30 to 54%':
    dataset_cmpn.iloc[i,18] = 2
  if dataset_cmpn.iloc[i,18] == 'Below 30%':
    dataset_cmpn.iloc[i,18] = 1

  if dataset_cmpn.iloc[i,19] == 'Thoroughly':
    dataset_cmpn.iloc[i,19] = 5
  if dataset_cmpn.iloc[i,19] == 'Satisfactorily':
    dataset_cmpn.iloc[i,19] = 4
  if dataset_cmpn.iloc[i,19] == 'Poorly':
    dataset_cmpn.iloc[i,19] = 3
  if dataset_cmpn.iloc[i,19] == 'Indifferently':
    dataset_cmpn.iloc[i,19] = 2
  if dataset_cmpn.iloc[i,19] == 'Won’t teach at all':
    dataset_cmpn.iloc[i,19] = 1

  if dataset_cmpn.iloc[i,20] == 'Always effective':
    dataset_cmpn.iloc[i,20] = 5
  if dataset_cmpn.iloc[i,20] == 'Sometimes effective':
    dataset_cmpn.iloc[i,20] = 4
  if dataset_cmpn.iloc[i,20] == 'Just satisfactorily':
    dataset_cmpn.iloc[i,20] = 3
  if dataset_cmpn.iloc[i,20] == 'Generally ineffective':
    dataset_cmpn.iloc[i,20] = 2
  if dataset_cmpn.iloc[i,20] == 'Very poor communication':
    dataset_cmpn.iloc[i,20] = 1

  if dataset_cmpn.iloc[i,21] == 'Excellent':
    dataset_cmpn.iloc[i,21] = 5
  if dataset_cmpn.iloc[i,21] == 'Very good':
    dataset_cmpn.iloc[i,21] = 4
  if dataset_cmpn.iloc[i,21] == 'Good':
    dataset_cmpn.iloc[i,21] = 3
  if dataset_cmpn.iloc[i,21] == 'Fair':
    dataset_cmpn.iloc[i,21] = 2
  if dataset_cmpn.iloc[i,21] == 'Poor':
    dataset_cmpn.iloc[i,21] = 1

  if dataset_cmpn.iloc[i,22] == 'Always fair':
    dataset_cmpn.iloc[i,22] = 5
  if dataset_cmpn.iloc[i,22] == 'Usually fair':
    dataset_cmpn.iloc[i,22] = 4
  if dataset_cmpn.iloc[i,22] == 'Sometimes unfair':
    dataset_cmpn.iloc[i,22] = 3
  if dataset_cmpn.iloc[i,22] == 'Usually unfair':
    dataset_cmpn.iloc[i,22] = 2
  if dataset_cmpn.iloc[i,22] == 'Unfair':
    dataset_cmpn.iloc[i,22] = 1

  if dataset_cmpn.iloc[i,23] == 'Every time':
    dataset_cmpn.iloc[i,23] = 5
  if dataset_cmpn.iloc[i,23] == 'Usually':
    dataset_cmpn.iloc[i,23] = 4
  if dataset_cmpn.iloc[i,23] == 'Occasionally/Sometimes':
    dataset_cmpn.iloc[i,23] = 3
  if dataset_cmpn.iloc[i,23] == 'Rarely':
    dataset_cmpn.iloc[i,23] = 2
  if dataset_cmpn.iloc[i,23] == 'Never':
    dataset_cmpn.iloc[i,23] = 1

  if dataset_cmpn.iloc[i,24] == 'Regularly':
    dataset_cmpn.iloc[i,24] = 5
  if dataset_cmpn.iloc[i,24] == 'Often':
    dataset_cmpn.iloc[i,24] = 4
  if dataset_cmpn.iloc[i,24] == 'Sometimes':
    dataset_cmpn.iloc[i,24] = 3
  if dataset_cmpn.iloc[i,24] == 'Rarely':
    dataset_cmpn.iloc[i,24] = 2
  if dataset_cmpn.iloc[i,24] == 'Never':
    dataset_cmpn.iloc[i,24] = 1

  if dataset_cmpn.iloc[i,25] == 'Significantly':
    dataset_cmpn.iloc[i,25] = 5
  if dataset_cmpn.iloc[i,25] == 'Very well':
    dataset_cmpn.iloc[i,25] = 4
  if dataset_cmpn.iloc[i,25] == 'Moderately':
    dataset_cmpn.iloc[i,25] = 3
  if dataset_cmpn.iloc[i,25] == 'Marginally':
    dataset_cmpn.iloc[i,25] = 2
  if dataset_cmpn.iloc[i,25] == 'Not at all':
    dataset_cmpn.iloc[i,25] = 1

  if dataset_cmpn.iloc[i,26] == 'Strongly agree':
    dataset_cmpn.iloc[i,26] = 5
  if dataset_cmpn.iloc[i,26] == 'Agree':
    dataset_cmpn.iloc[i,26] = 4
  if dataset_cmpn.iloc[i,26] == 'Neutral':
    dataset_cmpn.iloc[i,26] = 3
  if dataset_cmpn.iloc[i,26] == 'Disagree':
    dataset_cmpn.iloc[i,26] = 2
  if dataset_cmpn.iloc[i,26] == 'Strongly Disagree':
    dataset_cmpn.iloc[i,26] = 1

  if dataset_cmpn.iloc[i,27] == 'Every time':
    dataset_cmpn.iloc[i,27] = 5
  if dataset_cmpn.iloc[i,27] == 'Usually':
    dataset_cmpn.iloc[i,27] = 4
  if dataset_cmpn.iloc[i,27] == 'Occasionally/Sometimes':
    dataset_cmpn.iloc[i,27] = 3
  if dataset_cmpn.iloc[i,27] == 'Rarely':
    dataset_cmpn.iloc[i,27] = 2
  if dataset_cmpn.iloc[i,27] == 'Never':
    dataset_cmpn.iloc[i,27] = 1

  if dataset_cmpn.iloc[i,28] == 'Every time':
    dataset_cmpn.iloc[i,28] = 5
  if dataset_cmpn.iloc[i,28] == 'Usually':
    dataset_cmpn.iloc[i,28] = 4
  if dataset_cmpn.iloc[i,28] == 'Occasionally/Sometimes':
    dataset_cmpn.iloc[i,28] = 3
  if dataset_cmpn.iloc[i,28] == 'Rarely':
    dataset_cmpn.iloc[i,28] = 2
  if dataset_cmpn.iloc[i,28] == 'Never':
    dataset_cmpn.iloc[i,28] = 1

  if dataset_cmpn.iloc[i,29] == 'Every time':
    dataset_cmpn.iloc[i,29] = 5
  if dataset_cmpn.iloc[i,29] == 'Usually':
    dataset_cmpn.iloc[i,29] = 4
  if dataset_cmpn.iloc[i,29] == 'Occasionally/Sometimes':
    dataset_cmpn.iloc[i,29] = 3
  if dataset_cmpn.iloc[i,29] == 'Rarely':
    dataset_cmpn.iloc[i,29] = 2
  if dataset_cmpn.iloc[i,29] == 'Never':
    dataset_cmpn.iloc[i,29] = 1

  if dataset_cmpn.iloc[i,30] == 'Fully':
    dataset_cmpn.iloc[i,30] = 5
  if dataset_cmpn.iloc[i,30] == 'Reasonably':
    dataset_cmpn.iloc[i,30] = 4
  if dataset_cmpn.iloc[i,30] == 'Partially':
    dataset_cmpn.iloc[i,30] = 3
  if dataset_cmpn.iloc[i,30] == 'Slightly':
    dataset_cmpn.iloc[i,30] = 2
  if dataset_cmpn.iloc[i,30] == 'Unable to':
    dataset_cmpn.iloc[i,30] = 1

  if dataset_cmpn.iloc[i,31] == 'Every time':
    dataset_cmpn.iloc[i,31] = 5
  if dataset_cmpn.iloc[i,31] == 'Usually':
    dataset_cmpn.iloc[i,31] = 4
  if dataset_cmpn.iloc[i,31] == 'Occasionally/Sometimes':
    dataset_cmpn.iloc[i,31] = 3
  if dataset_cmpn.iloc[i,31] == 'Rarely':
    dataset_cmpn.iloc[i,31] = 2
  if dataset_cmpn.iloc[i,31] == 'Never':
    dataset_cmpn.iloc[i,31] = 1

  if dataset_cmpn.iloc[i,32] == 'Strongly agree':
    dataset_cmpn.iloc[i,32] = 5
  if dataset_cmpn.iloc[i,32] == 'Agree':
    dataset_cmpn.iloc[i,32] = 4
  if dataset_cmpn.iloc[i,32] == 'Neutral':
    dataset_cmpn.iloc[i,32] = 3
  if dataset_cmpn.iloc[i,32] == 'Disagree':
    dataset_cmpn.iloc[i,32] = 2
  if dataset_cmpn.iloc[i,32] == 'Strongly Disagree':
    dataset_cmpn.iloc[i,32] = 1

  if dataset_cmpn.iloc[i,33] == 'To a great extent':
    dataset_cmpn.iloc[i,33] = 5
  if dataset_cmpn.iloc[i,33] == 'Moderate':
    dataset_cmpn.iloc[i,33] = 4
  if dataset_cmpn.iloc[i,33] == 'Some what':
    dataset_cmpn.iloc[i,33] = 3
  if dataset_cmpn.iloc[i,33] == 'Very little':
    dataset_cmpn.iloc[i,33] = 2
  if dataset_cmpn.iloc[i,33] == 'Not at all':
    dataset_cmpn.iloc[i,33] = 1

  if dataset_cmpn.iloc[i,34] == 'Strongly agree':
    dataset_cmpn.iloc[i,34] = 5
  if dataset_cmpn.iloc[i,34] == 'Agree':
    dataset_cmpn.iloc[i,34] = 4
  if dataset_cmpn.iloc[i,34] == 'Neutral':
    dataset_cmpn.iloc[i,34] = 3
  if dataset_cmpn.iloc[i,34] == 'Disagree':
    dataset_cmpn.iloc[i,34] = 2
  if dataset_cmpn.iloc[i,34] == 'Strongly Disagree':
    dataset_cmpn.iloc[i,34] = 1

  if dataset_cmpn.iloc[i,35] == 'To a great extent':
    dataset_cmpn.iloc[i,35] = 5
  if dataset_cmpn.iloc[i,35] == 'Moderate':
    dataset_cmpn.iloc[i,35] = 4
  if dataset_cmpn.iloc[i,35] == 'Some what':
    dataset_cmpn.iloc[i,35] = 3
  if dataset_cmpn.iloc[i,35] == 'Very little':
    dataset_cmpn.iloc[i,35] = 2
  if dataset_cmpn.iloc[i,35] == 'Not at all':
    dataset_cmpn.iloc[i,35] = 1

  if dataset_cmpn.iloc[i,36] == 'Above 90%':
    dataset_cmpn.iloc[i,36] = 5
  if dataset_cmpn.iloc[i,36] == '70–89%':
    dataset_cmpn.iloc[i,36] = 4
  if dataset_cmpn.iloc[i,36] == '50–69%':
    dataset_cmpn.iloc[i,36] = 3
  if dataset_cmpn.iloc[i,36] == '30–49%':
    dataset_cmpn.iloc[i,36] = 2
  if dataset_cmpn.iloc[i,36] == 'Below 29%':
    dataset_cmpn.iloc[i,36] = 1

  if dataset_cmpn.iloc[i,37] == 'Strongly agree':
    dataset_cmpn.iloc[i,37] = 5
  if dataset_cmpn.iloc[i,37] == 'Agree':
    dataset_cmpn.iloc[i,37] = 4
  if dataset_cmpn.iloc[i,37] == 'Neutral':
    dataset_cmpn.iloc[i,37] = 3
  if dataset_cmpn.iloc[i,37] == 'Disagree':
    dataset_cmpn.iloc[i,37] = 2
  if dataset_cmpn.iloc[i,37] == 'Strongly Disagree':
    dataset_cmpn.iloc[i,37] = 1

for i in range(0,n_inft):
  if dataset_inft.iloc[i,18] == '85 to 100%':
    dataset_inft.iloc[i,18] = 5
  if dataset_inft.iloc[i,18] == '70 to 84%':
    dataset_inft.iloc[i,18] = 4
  if dataset_inft.iloc[i,18] == '55 to 69%':
    dataset_inft.iloc[i,18] = 3
  if dataset_inft.iloc[i,18] == '30 to 54%':
    dataset_inft.iloc[i,18] = 2
  if dataset_inft.iloc[i,18] == 'Below 30%':
    dataset_inft.iloc[i,18] = 1

  if dataset_inft.iloc[i,19] == 'Thoroughly':
    dataset_inft.iloc[i,19] = 5
  if dataset_inft.iloc[i,19] == 'Satisfactorily':
    dataset_inft.iloc[i,19] = 4
  if dataset_inft.iloc[i,19] == 'Poorly':
    dataset_inft.iloc[i,19] = 3
  if dataset_inft.iloc[i,19] == 'Indifferently':
    dataset_inft.iloc[i,19] = 2
  if dataset_inft.iloc[i,19] == 'Won’t teach at all':
    dataset_inft.iloc[i,19] = 1

  if dataset_inft.iloc[i,20] == 'Always effective':
    dataset_inft.iloc[i,20] = 5
  if dataset_inft.iloc[i,20] == 'Sometimes effective':
    dataset_inft.iloc[i,20] = 4
  if dataset_inft.iloc[i,20] == 'Just satisfactorily':
    dataset_inft.iloc[i,20] = 3
  if dataset_inft.iloc[i,20] == 'Generally ineffective':
    dataset_inft.iloc[i,20] = 2
  if dataset_inft.iloc[i,20] == 'Very poor communication':
    dataset_inft.iloc[i,20] = 1

  if dataset_inft.iloc[i,21] == 'Excellent':
    dataset_inft.iloc[i,21] = 5
  if dataset_inft.iloc[i,21] == 'Very good':
    dataset_inft.iloc[i,21] = 4
  if dataset_inft.iloc[i,21] == 'Good':
    dataset_inft.iloc[i,21] = 3
  if dataset_inft.iloc[i,21] == 'Fair':
    dataset_inft.iloc[i,21] = 2
  if dataset_inft.iloc[i,21] == 'Poor':
    dataset_inft.iloc[i,21] = 1

  if dataset_inft.iloc[i,22] == 'Always fair':
    dataset_inft.iloc[i,22] = 5
  if dataset_inft.iloc[i,22] == 'Usually fair':
    dataset_inft.iloc[i,22] = 4
  if dataset_inft.iloc[i,22] == 'Sometimes unfair':
    dataset_inft.iloc[i,22] = 3
  if dataset_inft.iloc[i,22] == 'Usually unfair':
    dataset_inft.iloc[i,22] = 2
  if dataset_inft.iloc[i,22] == 'Unfair':
    dataset_inft.iloc[i,22] = 1

  if dataset_inft.iloc[i,23] == 'Every time':
    dataset_inft.iloc[i,23] = 5
  if dataset_inft.iloc[i,23] == 'Usually':
    dataset_inft.iloc[i,23] = 4
  if dataset_inft.iloc[i,23] == 'Occasionally/Sometimes':
    dataset_inft.iloc[i,23] = 3
  if dataset_inft.iloc[i,23] == 'Rarely':
    dataset_inft.iloc[i,23] = 2
  if dataset_inft.iloc[i,23] == 'Never':
    dataset_inft.iloc[i,23] = 1

  if dataset_inft.iloc[i,24] == 'Regularly':
    dataset_inft.iloc[i,24] = 5
  if dataset_inft.iloc[i,24] == 'Often':
    dataset_inft.iloc[i,24] = 4
  if dataset_inft.iloc[i,24] == 'Sometimes':
    dataset_inft.iloc[i,24] = 3
  if dataset_inft.iloc[i,24] == 'Rarely':
    dataset_inft.iloc[i,24] = 2
  if dataset_inft.iloc[i,24] == 'Never':
    dataset_inft.iloc[i,24] = 1

  if dataset_inft.iloc[i,25] == 'Significantly':
    dataset_inft.iloc[i,25] = 5
  if dataset_inft.iloc[i,25] == 'Very well':
    dataset_inft.iloc[i,25] = 4
  if dataset_inft.iloc[i,25] == 'Moderately':
    dataset_inft.iloc[i,25] = 3
  if dataset_inft.iloc[i,25] == 'Marginally':
    dataset_inft.iloc[i,25] = 2
  if dataset_inft.iloc[i,25] == 'Not at all':
    dataset_inft.iloc[i,25] = 1

  if dataset_inft.iloc[i,26] == 'Strongly agree':
    dataset_inft.iloc[i,26] = 5
  if dataset_inft.iloc[i,26] == 'Agree':
    dataset_inft.iloc[i,26] = 4
  if dataset_inft.iloc[i,26] == 'Neutral':
    dataset_inft.iloc[i,26] = 3
  if dataset_inft.iloc[i,26] == 'Disagree':
    dataset_inft.iloc[i,26] = 2
  if dataset_inft.iloc[i,26] == 'Strongly Disagree':
    dataset_inft.iloc[i,26] = 1

  if dataset_inft.iloc[i,27] == 'Every time':
    dataset_inft.iloc[i,27] = 5
  if dataset_inft.iloc[i,27] == 'Usually':
    dataset_inft.iloc[i,27] = 4
  if dataset_inft.iloc[i,27] == 'Occasionally/Sometimes':
    dataset_inft.iloc[i,27] = 3
  if dataset_inft.iloc[i,27] == 'Rarely':
    dataset_inft.iloc[i,27] = 2
  if dataset_inft.iloc[i,27] == 'Never':
    dataset_inft.iloc[i,27] = 1

  if dataset_inft.iloc[i,28] == 'Every time':
    dataset_inft.iloc[i,28] = 5
  if dataset_inft.iloc[i,28] == 'Usually':
    dataset_inft.iloc[i,28] = 4
  if dataset_inft.iloc[i,28] == 'Occasionally/Sometimes':
    dataset_inft.iloc[i,28] = 3
  if dataset_inft.iloc[i,28] == 'Rarely':
    dataset_inft.iloc[i,28] = 2
  if dataset_inft.iloc[i,28] == 'Never':
    dataset_inft.iloc[i,28] = 1

  if dataset_inft.iloc[i,29] == 'Every time':
    dataset_inft.iloc[i,29] = 5
  if dataset_inft.iloc[i,29] == 'Usually':
    dataset_inft.iloc[i,29] = 4
  if dataset_inft.iloc[i,29] == 'Occasionally/Sometimes':
    dataset_inft.iloc[i,29] = 3
  if dataset_inft.iloc[i,29] == 'Rarely':
    dataset_inft.iloc[i,29] = 2
  if dataset_inft.iloc[i,29] == 'Never':
    dataset_inft.iloc[i,29] = 1

  if dataset_inft.iloc[i,30] == 'Fully':
    dataset_inft.iloc[i,30] = 5
  if dataset_inft.iloc[i,30] == 'Reasonably':
    dataset_inft.iloc[i,30] = 4
  if dataset_inft.iloc[i,30] == 'Partially':
    dataset_inft.iloc[i,30] = 3
  if dataset_inft.iloc[i,30] == 'Slightly':
    dataset_inft.iloc[i,30] = 2
  if dataset_inft.iloc[i,30] == 'Unable to':
    dataset_inft.iloc[i,30] = 1

  if dataset_inft.iloc[i,31] == 'Every time':
    dataset_inft.iloc[i,31] = 5
  if dataset_inft.iloc[i,31] == 'Usually':
    dataset_inft.iloc[i,31] = 4
  if dataset_inft.iloc[i,31] == 'Occasionally/Sometimes':
    dataset_inft.iloc[i,31] = 3
  if dataset_inft.iloc[i,31] == 'Rarely':
    dataset_inft.iloc[i,31] = 2
  if dataset_inft.iloc[i,31] == 'Never':
    dataset_inft.iloc[i,31] = 1

  if dataset_inft.iloc[i,32] == 'Strongly agree':
    dataset_inft.iloc[i,32] = 5
  if dataset_inft.iloc[i,32] == 'Agree':
    dataset_inft.iloc[i,32] = 4
  if dataset_inft.iloc[i,32] == 'Neutral':
    dataset_inft.iloc[i,32] = 3
  if dataset_inft.iloc[i,32] == 'Disagree':
    dataset_inft.iloc[i,32] = 2
  if dataset_inft.iloc[i,32] == 'Strongly Disagree':
    dataset_inft.iloc[i,32] = 1

  if dataset_inft.iloc[i,33] == 'To a great extent':
    dataset_inft.iloc[i,33] = 5
  if dataset_inft.iloc[i,33] == 'Moderate':
    dataset_inft.iloc[i,33] = 4
  if dataset_inft.iloc[i,33] == 'Some what':
    dataset_inft.iloc[i,33] = 3
  if dataset_inft.iloc[i,33] == 'Very little':
    dataset_inft.iloc[i,33] = 2
  if dataset_inft.iloc[i,33] == 'Not at all':
    dataset_inft.iloc[i,33] = 1

  if dataset_inft.iloc[i,34] == 'Strongly agree':
    dataset_inft.iloc[i,34] = 5
  if dataset_inft.iloc[i,34] == 'Agree':
    dataset_inft.iloc[i,34] = 4
  if dataset_inft.iloc[i,34] == 'Neutral':
    dataset_inft.iloc[i,34] = 3
  if dataset_inft.iloc[i,34] == 'Disagree':
    dataset_inft.iloc[i,34] = 2
  if dataset_inft.iloc[i,34] == 'Strongly Disagree':
    dataset_inft.iloc[i,34] = 1

  if dataset_inft.iloc[i,35] == 'To a great extent':
    dataset_inft.iloc[i,35] = 5
  if dataset_inft.iloc[i,35] == 'Moderate':
    dataset_inft.iloc[i,35] = 4
  if dataset_inft.iloc[i,35] == 'Some what':
    dataset_inft.iloc[i,35] = 3
  if dataset_inft.iloc[i,35] == 'Very little':
    dataset_inft.iloc[i,35] = 2
  if dataset_inft.iloc[i,35] == 'Not at all':
    dataset_inft.iloc[i,35] = 1

  if dataset_inft.iloc[i,36] == 'Above 90%':
    dataset_inft.iloc[i,36] = 5
  if dataset_inft.iloc[i,36] == '70–89%':
    dataset_inft.iloc[i,36] = 4
  if dataset_inft.iloc[i,36] == '50–69%':
    dataset_inft.iloc[i,36] = 3
  if dataset_inft.iloc[i,36] == '30–49%':
    dataset_inft.iloc[i,36] = 2
  if dataset_inft.iloc[i,36] == 'Below 29%':
    dataset_inft.iloc[i,36] = 1

  if dataset_inft.iloc[i,37] == 'Strongly agree':
    dataset_inft.iloc[i,37] = 5
  if dataset_inft.iloc[i,37] == 'Agree':
    dataset_inft.iloc[i,37] = 4
  if dataset_inft.iloc[i,37] == 'Neutral':
    dataset_inft.iloc[i,37] = 3
  if dataset_inft.iloc[i,37] == 'Disagree':
    dataset_inft.iloc[i,37] = 2
  if dataset_inft.iloc[i,37] == 'Strongly Disagree':
    dataset_inft.iloc[i,37] = 1

for i in range(0,n_etrx):
  if dataset_etrx.iloc[i,18] == '85 to 100%':
    dataset_etrx.iloc[i,18] = 5
  if dataset_etrx.iloc[i,18] == '70 to 84%':
    dataset_etrx.iloc[i,18] = 4
  if dataset_etrx.iloc[i,18] == '55 to 69%':
    dataset_etrx.iloc[i,18] = 3
  if dataset_etrx.iloc[i,18] == '30 to 54%':
    dataset_etrx.iloc[i,18] = 2
  if dataset_etrx.iloc[i,18] == 'Below 30%':
    dataset_etrx.iloc[i,18] = 1

  if dataset_etrx.iloc[i,19] == 'Thoroughly':
    dataset_etrx.iloc[i,19] = 5
  if dataset_etrx.iloc[i,19] == 'Satisfactorily':
    dataset_etrx.iloc[i,19] = 4
  if dataset_etrx.iloc[i,19] == 'Poorly':
    dataset_etrx.iloc[i,19] = 3
  if dataset_etrx.iloc[i,19] == 'Indifferently':
    dataset_etrx.iloc[i,19] = 2
  if dataset_etrx.iloc[i,19] == 'Won’t teach at all':
    dataset_etrx.iloc[i,19] = 1

  if dataset_etrx.iloc[i,20] == 'Always effective':
    dataset_etrx.iloc[i,20] = 5
  if dataset_etrx.iloc[i,20] == 'Sometimes effective':
    dataset_etrx.iloc[i,20] = 4
  if dataset_etrx.iloc[i,20] == 'Just satisfactorily':
    dataset_etrx.iloc[i,20] = 3
  if dataset_etrx.iloc[i,20] == 'Generally ineffective':
    dataset_etrx.iloc[i,20] = 2
  if dataset_etrx.iloc[i,20] == 'Very poor communication':
    dataset_etrx.iloc[i,20] = 1

  if dataset_etrx.iloc[i,21] == 'Excellent':
    dataset_etrx.iloc[i,21] = 5
  if dataset_etrx.iloc[i,21] == 'Very good':
    dataset_etrx.iloc[i,21] = 4
  if dataset_etrx.iloc[i,21] == 'Good':
    dataset_etrx.iloc[i,21] = 3
  if dataset_etrx.iloc[i,21] == 'Fair':
    dataset_etrx.iloc[i,21] = 2
  if dataset_etrx.iloc[i,21] == 'Poor':
    dataset_etrx.iloc[i,21] = 1

  if dataset_etrx.iloc[i,22] == 'Always fair':
    dataset_etrx.iloc[i,22] = 5
  if dataset_etrx.iloc[i,22] == 'Usually fair':
    dataset_etrx.iloc[i,22] = 4
  if dataset_etrx.iloc[i,22] == 'Sometimes unfair':
    dataset_etrx.iloc[i,22] = 3
  if dataset_etrx.iloc[i,22] == 'Usually unfair':
    dataset_etrx.iloc[i,22] = 2
  if dataset_etrx.iloc[i,22] == 'Unfair':
    dataset_etrx.iloc[i,22] = 1

  if dataset_etrx.iloc[i,23] == 'Every time':
    dataset_etrx.iloc[i,23] = 5
  if dataset_etrx.iloc[i,23] == 'Usually':
    dataset_etrx.iloc[i,23] = 4
  if dataset_etrx.iloc[i,23] == 'Occasionally/Sometimes':
    dataset_etrx.iloc[i,23] = 3
  if dataset_etrx.iloc[i,23] == 'Rarely':
    dataset_etrx.iloc[i,23] = 2
  if dataset_etrx.iloc[i,23] == 'Never':
    dataset_etrx.iloc[i,23] = 1

  if dataset_etrx.iloc[i,24] == 'Regularly':
    dataset_etrx.iloc[i,24] = 5
  if dataset_etrx.iloc[i,24] == 'Often':
    dataset_etrx.iloc[i,24] = 4
  if dataset_etrx.iloc[i,24] == 'Sometimes':
    dataset_etrx.iloc[i,24] = 3
  if dataset_etrx.iloc[i,24] == 'Rarely':
    dataset_etrx.iloc[i,24] = 2
  if dataset_etrx.iloc[i,24] == 'Never':
    dataset_etrx.iloc[i,24] = 1

  if dataset_etrx.iloc[i,25] == 'Significantly':
    dataset_etrx.iloc[i,25] = 5
  if dataset_etrx.iloc[i,25] == 'Very well':
    dataset_etrx.iloc[i,25] = 4
  if dataset_etrx.iloc[i,25] == 'Moderately':
    dataset_etrx.iloc[i,25] = 3
  if dataset_etrx.iloc[i,25] == 'Marginally':
    dataset_etrx.iloc[i,25] = 2
  if dataset_etrx.iloc[i,25] == 'Not at all':
    dataset_etrx.iloc[i,25] = 1

  if dataset_etrx.iloc[i,26] == 'Strongly agree':
    dataset_etrx.iloc[i,26] = 5
  if dataset_etrx.iloc[i,26] == 'Agree':
    dataset_etrx.iloc[i,26] = 4
  if dataset_etrx.iloc[i,26] == 'Neutral':
    dataset_etrx.iloc[i,26] = 3
  if dataset_etrx.iloc[i,26] == 'Disagree':
    dataset_etrx.iloc[i,26] = 2
  if dataset_etrx.iloc[i,26] == 'Strongly Disagree':
    dataset_etrx.iloc[i,26] = 1

  if dataset_etrx.iloc[i,27] == 'Every time':
    dataset_etrx.iloc[i,27] = 5
  if dataset_etrx.iloc[i,27] == 'Usually':
    dataset_etrx.iloc[i,27] = 4
  if dataset_etrx.iloc[i,27] == 'Occasionally/Sometimes':
    dataset_etrx.iloc[i,27] = 3
  if dataset_etrx.iloc[i,27] == 'Rarely':
    dataset_etrx.iloc[i,27] = 2
  if dataset_etrx.iloc[i,27] == 'Never':
    dataset_etrx.iloc[i,27] = 1

  if dataset_etrx.iloc[i,28] == 'Every time':
    dataset_etrx.iloc[i,28] = 5
  if dataset_etrx.iloc[i,28] == 'Usually':
    dataset_etrx.iloc[i,28] = 4
  if dataset_etrx.iloc[i,28] == 'Occasionally/Sometimes':
    dataset_etrx.iloc[i,28] = 3
  if dataset_etrx.iloc[i,28] == 'Rarely':
    dataset_etrx.iloc[i,28] = 2
  if dataset_etrx.iloc[i,28] == 'Never':
    dataset_etrx.iloc[i,28] = 1

  if dataset_etrx.iloc[i,29] == 'Every time':
    dataset_etrx.iloc[i,29] = 5
  if dataset_etrx.iloc[i,29] == 'Usually':
    dataset_etrx.iloc[i,29] = 4
  if dataset_etrx.iloc[i,29] == 'Occasionally/Sometimes':
    dataset_etrx.iloc[i,29] = 3
  if dataset_etrx.iloc[i,29] == 'Rarely':
    dataset_etrx.iloc[i,29] = 2
  if dataset_etrx.iloc[i,29] == 'Never':
    dataset_etrx.iloc[i,29] = 1

  if dataset_etrx.iloc[i,30] == 'Fully':
    dataset_etrx.iloc[i,30] = 5
  if dataset_etrx.iloc[i,30] == 'Reasonably':
    dataset_etrx.iloc[i,30] = 4
  if dataset_etrx.iloc[i,30] == 'Partially':
    dataset_etrx.iloc[i,30] = 3
  if dataset_etrx.iloc[i,30] == 'Slightly':
    dataset_etrx.iloc[i,30] = 2
  if dataset_etrx.iloc[i,30] == 'Unable to':
    dataset_etrx.iloc[i,30] = 1

  if dataset_etrx.iloc[i,31] == 'Every time':
    dataset_etrx.iloc[i,31] = 5
  if dataset_etrx.iloc[i,31] == 'Usually':
    dataset_etrx.iloc[i,31] = 4
  if dataset_etrx.iloc[i,31] == 'Occasionally/Sometimes':
    dataset_etrx.iloc[i,31] = 3
  if dataset_etrx.iloc[i,31] == 'Rarely':
    dataset_etrx.iloc[i,31] = 2
  if dataset_etrx.iloc[i,31] == 'Never':
    dataset_etrx.iloc[i,31] = 1

  if dataset_etrx.iloc[i,32] == 'Strongly agree':
    dataset_etrx.iloc[i,32] = 5
  if dataset_etrx.iloc[i,32] == 'Agree':
    dataset_etrx.iloc[i,32] = 4
  if dataset_etrx.iloc[i,32] == 'Neutral':
    dataset_etrx.iloc[i,32] = 3
  if dataset_etrx.iloc[i,32] == 'Disagree':
    dataset_etrx.iloc[i,32] = 2
  if dataset_etrx.iloc[i,32] == 'Strongly Disagree':
    dataset_etrx.iloc[i,32] = 1

  if dataset_etrx.iloc[i,33] == 'To a great extent':
    dataset_etrx.iloc[i,33] = 5
  if dataset_etrx.iloc[i,33] == 'Moderate':
    dataset_etrx.iloc[i,33] = 4
  if dataset_etrx.iloc[i,33] == 'Some what':
    dataset_etrx.iloc[i,33] = 3
  if dataset_etrx.iloc[i,33] == 'Very little':
    dataset_etrx.iloc[i,33] = 2
  if dataset_etrx.iloc[i,33] == 'Not at all':
    dataset_etrx.iloc[i,33] = 1

  if dataset_etrx.iloc[i,34] == 'Strongly agree':
    dataset_etrx.iloc[i,34] = 5
  if dataset_etrx.iloc[i,34] == 'Agree':
    dataset_etrx.iloc[i,34] = 4
  if dataset_etrx.iloc[i,34] == 'Neutral':
    dataset_etrx.iloc[i,34] = 3
  if dataset_etrx.iloc[i,34] == 'Disagree':
    dataset_etrx.iloc[i,34] = 2
  if dataset_etrx.iloc[i,34] == 'Strongly Disagree':
    dataset_etrx.iloc[i,34] = 1

  if dataset_etrx.iloc[i,35] == 'To a great extent':
    dataset_etrx.iloc[i,35] = 5
  if dataset_etrx.iloc[i,35] == 'Moderate':
    dataset_etrx.iloc[i,35] = 4
  if dataset_etrx.iloc[i,35] == 'Some what':
    dataset_etrx.iloc[i,35] = 3
  if dataset_etrx.iloc[i,35] == 'Very little':
    dataset_etrx.iloc[i,35] = 2
  if dataset_etrx.iloc[i,35] == 'Not at all':
    dataset_etrx.iloc[i,35] = 1

  if dataset_etrx.iloc[i,36] == 'Above 90%':
    dataset_etrx.iloc[i,36] = 5
  if dataset_etrx.iloc[i,36] == '70–89%':
    dataset_etrx.iloc[i,36] = 4
  if dataset_etrx.iloc[i,36] == '50–69%':
    dataset_etrx.iloc[i,36] = 3
  if dataset_etrx.iloc[i,36] == '30–49%':
    dataset_etrx.iloc[i,36] = 2
  if dataset_etrx.iloc[i,36] == 'Below 29%':
    dataset_etrx.iloc[i,36] = 1

  if dataset_etrx.iloc[i,37] == 'Strongly agree':
    dataset_etrx.iloc[i,37] = 5
  if dataset_etrx.iloc[i,37] == 'Agree':
    dataset_etrx.iloc[i,37] = 4
  if dataset_etrx.iloc[i,37] == 'Neutral':
    dataset_etrx.iloc[i,37] = 3
  if dataset_etrx.iloc[i,37] == 'Disagree':
    dataset_etrx.iloc[i,37] = 2
  if dataset_etrx.iloc[i,37] == 'Strongly Disagree':
    dataset_etrx.iloc[i,37] = 1

for i in range(0,n_extc):
  if dataset_extc.iloc[i,18] == '85 to 100%':
    dataset_extc.iloc[i,18] = 5
  if dataset_extc.iloc[i,18] == '70 to 84%':
    dataset_extc.iloc[i,18] = 4
  if dataset_extc.iloc[i,18] == '55 to 69%':
    dataset_extc.iloc[i,18] = 3
  if dataset_extc.iloc[i,18] == '30 to 54%':
    dataset_extc.iloc[i,18] = 2
  if dataset_extc.iloc[i,18] == 'Below 30%':
    dataset_extc.iloc[i,18] = 1

  if dataset_extc.iloc[i,19] == 'Thoroughly':
    dataset_extc.iloc[i,19] = 5
  if dataset_extc.iloc[i,19] == 'Satisfactorily':
    dataset_extc.iloc[i,19] = 4
  if dataset_extc.iloc[i,19] == 'Poorly':
    dataset_extc.iloc[i,19] = 3
  if dataset_extc.iloc[i,19] == 'Indifferently':
    dataset_extc.iloc[i,19] = 2
  if dataset_extc.iloc[i,19] == 'Won’t teach at all':
    dataset_extc.iloc[i,19] = 1

  if dataset_extc.iloc[i,20] == 'Always effective':
    dataset_extc.iloc[i,20] = 5
  if dataset_extc.iloc[i,20] == 'Sometimes effective':
    dataset_extc.iloc[i,20] = 4
  if dataset_extc.iloc[i,20] == 'Just satisfactorily':
    dataset_extc.iloc[i,20] = 3
  if dataset_extc.iloc[i,20] == 'Generally ineffective':
    dataset_extc.iloc[i,20] = 2
  if dataset_extc.iloc[i,20] == 'Very poor communication':
    dataset_extc.iloc[i,20] = 1

  if dataset_extc.iloc[i,21] == 'Excellent':
    dataset_extc.iloc[i,21] = 5
  if dataset_extc.iloc[i,21] == 'Very good':
    dataset_extc.iloc[i,21] = 4
  if dataset_extc.iloc[i,21] == 'Good':
    dataset_extc.iloc[i,21] = 3
  if dataset_extc.iloc[i,21] == 'Fair':
    dataset_extc.iloc[i,21] = 2
  if dataset_extc.iloc[i,21] == 'Poor':
    dataset_extc.iloc[i,21] = 1

  if dataset_extc.iloc[i,22] == 'Always fair':
    dataset_extc.iloc[i,22] = 5
  if dataset_extc.iloc[i,22] == 'Usually fair':
    dataset_extc.iloc[i,22] = 4
  if dataset_extc.iloc[i,22] == 'Sometimes unfair':
    dataset_extc.iloc[i,22] = 3
  if dataset_extc.iloc[i,22] == 'Usually unfair':
    dataset_extc.iloc[i,22] = 2
  if dataset_extc.iloc[i,22] == 'Unfair':
    dataset_extc.iloc[i,22] = 1

  if dataset_extc.iloc[i,23] == 'Every time':
    dataset_extc.iloc[i,23] = 5
  if dataset_extc.iloc[i,23] == 'Usually':
    dataset_extc.iloc[i,23] = 4
  if dataset_extc.iloc[i,23] == 'Occasionally/Sometimes':
    dataset_extc.iloc[i,23] = 3
  if dataset_extc.iloc[i,23] == 'Rarely':
    dataset_extc.iloc[i,23] = 2
  if dataset_extc.iloc[i,23] == 'Never':
    dataset_extc.iloc[i,23] = 1

  if dataset_extc.iloc[i,24] == 'Regularly':
    dataset_extc.iloc[i,24] = 5
  if dataset_extc.iloc[i,24] == 'Often':
    dataset_extc.iloc[i,24] = 4
  if dataset_extc.iloc[i,24] == 'Sometimes':
    dataset_extc.iloc[i,24] = 3
  if dataset_extc.iloc[i,24] == 'Rarely':
    dataset_extc.iloc[i,24] = 2
  if dataset_extc.iloc[i,24] == 'Never':
    dataset_extc.iloc[i,24] = 1

  if dataset_extc.iloc[i,25] == 'Significantly':
    dataset_extc.iloc[i,25] = 5
  if dataset_extc.iloc[i,25] == 'Very well':
    dataset_extc.iloc[i,25] = 4
  if dataset_extc.iloc[i,25] == 'Moderately':
    dataset_extc.iloc[i,25] = 3
  if dataset_extc.iloc[i,25] == 'Marginally':
    dataset_extc.iloc[i,25] = 2
  if dataset_extc.iloc[i,25] == 'Not at all':
    dataset_extc.iloc[i,25] = 1

  if dataset_extc.iloc[i,26] == 'Strongly agree':
    dataset_extc.iloc[i,26] = 5
  if dataset_extc.iloc[i,26] == 'Agree':
    dataset_extc.iloc[i,26] = 4
  if dataset_extc.iloc[i,26] == 'Neutral':
    dataset_extc.iloc[i,26] = 3
  if dataset_extc.iloc[i,26] == 'Disagree':
    dataset_extc.iloc[i,26] = 2
  if dataset_extc.iloc[i,26] == 'Strongly Disagree':
    dataset_extc.iloc[i,26] = 1

  if dataset_extc.iloc[i,27] == 'Every time':
    dataset_extc.iloc[i,27] = 5
  if dataset_extc.iloc[i,27] == 'Usually':
    dataset_extc.iloc[i,27] = 4
  if dataset_extc.iloc[i,27] == 'Occasionally/Sometimes':
    dataset_extc.iloc[i,27] = 3
  if dataset_extc.iloc[i,27] == 'Rarely':
    dataset_extc.iloc[i,27] = 2
  if dataset_extc.iloc[i,27] == 'Never':
    dataset_extc.iloc[i,27] = 1

  if dataset_extc.iloc[i,28] == 'Every time':
    dataset_extc.iloc[i,28] = 5
  if dataset_extc.iloc[i,28] == 'Usually':
    dataset_extc.iloc[i,28] = 4
  if dataset_extc.iloc[i,28] == 'Occasionally/Sometimes':
    dataset_extc.iloc[i,28] = 3
  if dataset_extc.iloc[i,28] == 'Rarely':
    dataset_extc.iloc[i,28] = 2
  if dataset_extc.iloc[i,28] == 'Never':
    dataset_extc.iloc[i,28] = 1

  if dataset_extc.iloc[i,29] == 'Every time':
    dataset_extc.iloc[i,29] = 5
  if dataset_extc.iloc[i,29] == 'Usually':
    dataset_extc.iloc[i,29] = 4
  if dataset_extc.iloc[i,29] == 'Occasionally/Sometimes':
    dataset_extc.iloc[i,29] = 3
  if dataset_extc.iloc[i,29] == 'Rarely':
    dataset_extc.iloc[i,29] = 2
  if dataset_extc.iloc[i,29] == 'Never':
    dataset_extc.iloc[i,29] = 1

  if dataset_extc.iloc[i,30] == 'Fully':
    dataset_extc.iloc[i,30] = 5
  if dataset_extc.iloc[i,30] == 'Reasonably':
    dataset_extc.iloc[i,30] = 4
  if dataset_extc.iloc[i,30] == 'Partially':
    dataset_extc.iloc[i,30] = 3
  if dataset_extc.iloc[i,30] == 'Slightly':
    dataset_extc.iloc[i,30] = 2
  if dataset_extc.iloc[i,30] == 'Unable to':
    dataset_extc.iloc[i,30] = 1

  if dataset_extc.iloc[i,31] == 'Every time':
    dataset_extc.iloc[i,31] = 5
  if dataset_extc.iloc[i,31] == 'Usually':
    dataset_extc.iloc[i,31] = 4
  if dataset_extc.iloc[i,31] == 'Occasionally/Sometimes':
    dataset_extc.iloc[i,31] = 3
  if dataset_extc.iloc[i,31] == 'Rarely':
    dataset_extc.iloc[i,31] = 2
  if dataset_extc.iloc[i,31] == 'Never':
    dataset_extc.iloc[i,31] = 1

  if dataset_extc.iloc[i,32] == 'Strongly agree':
    dataset_extc.iloc[i,32] = 5
  if dataset_extc.iloc[i,32] == 'Agree':
    dataset_extc.iloc[i,32] = 4
  if dataset_extc.iloc[i,32] == 'Neutral':
    dataset_extc.iloc[i,32] = 3
  if dataset_extc.iloc[i,32] == 'Disagree':
    dataset_extc.iloc[i,32] = 2
  if dataset_extc.iloc[i,32] == 'Strongly Disagree':
    dataset_extc.iloc[i,32] = 1

  if dataset_extc.iloc[i,33] == 'To a great extent':
    dataset_extc.iloc[i,33] = 5
  if dataset_extc.iloc[i,33] == 'Moderate':
    dataset_extc.iloc[i,33] = 4
  if dataset_extc.iloc[i,33] == 'Some what':
    dataset_extc.iloc[i,33] = 3
  if dataset_extc.iloc[i,33] == 'Very little':
    dataset_extc.iloc[i,33] = 2
  if dataset_extc.iloc[i,33] == 'Not at all':
    dataset_extc.iloc[i,33] = 1

  if dataset_extc.iloc[i,34] == 'Strongly agree':
    dataset_extc.iloc[i,34] = 5
  if dataset_extc.iloc[i,34] == 'Agree':
    dataset_extc.iloc[i,34] = 4
  if dataset_extc.iloc[i,34] == 'Neutral':
    dataset_extc.iloc[i,34] = 3
  if dataset_extc.iloc[i,34] == 'Disagree':
    dataset_extc.iloc[i,34] = 2
  if dataset_extc.iloc[i,34] == 'Strongly Disagree':
    dataset_extc.iloc[i,34] = 1

  if dataset_extc.iloc[i,35] == 'To a great extent':
    dataset_extc.iloc[i,35] = 5
  if dataset_extc.iloc[i,35] == 'Moderate':
    dataset_extc.iloc[i,35] = 4
  if dataset_extc.iloc[i,35] == 'Some what':
    dataset_extc.iloc[i,35] = 3
  if dataset_extc.iloc[i,35] == 'Very little':
    dataset_extc.iloc[i,35] = 2
  if dataset_extc.iloc[i,35] == 'Not at all':
    dataset_extc.iloc[i,35] = 1

  if dataset_extc.iloc[i,36] == 'Above 90%':
    dataset_extc.iloc[i,36] = 5
  if dataset_extc.iloc[i,36] == '70–89%':
    dataset_extc.iloc[i,36] = 4
  if dataset_extc.iloc[i,36] == '50–69%':
    dataset_extc.iloc[i,36] = 3
  if dataset_extc.iloc[i,36] == '30–49%':
    dataset_extc.iloc[i,36] = 2
  if dataset_extc.iloc[i,36] == 'Below 29%':
    dataset_extc.iloc[i,36] = 1

  if dataset_extc.iloc[i,37] == 'Strongly agree':
    dataset_extc.iloc[i,37] = 5
  if dataset_extc.iloc[i,37] == 'Agree':
    dataset_extc.iloc[i,37] = 4
  if dataset_extc.iloc[i,37] == 'Neutral':
    dataset_extc.iloc[i,37] = 3
  if dataset_extc.iloc[i,37] == 'Disagree':
    dataset_extc.iloc[i,37] = 2
  if dataset_extc.iloc[i,37] == 'Strongly Disagree':
    dataset_extc.iloc[i,37] = 1

for i in range(0,n_biom):
  if dataset_biom.iloc[i,18] == '85 to 100%':
    dataset_biom.iloc[i,18] = 5
  if dataset_biom.iloc[i,18] == '70 to 84%':
    dataset_biom.iloc[i,18] = 4
  if dataset_biom.iloc[i,18] == '55 to 69%':
    dataset_biom.iloc[i,18] = 3
  if dataset_biom.iloc[i,18] == '30 to 54%':
    dataset_biom.iloc[i,18] = 2
  if dataset_biom.iloc[i,18] == 'Below 30%':
    dataset_biom.iloc[i,18] = 1

  if dataset_biom.iloc[i,19] == 'Thoroughly':
    dataset_biom.iloc[i,19] = 5
  if dataset_biom.iloc[i,19] == 'Satisfactorily':
    dataset_biom.iloc[i,19] = 4
  if dataset_biom.iloc[i,19] == 'Poorly':
    dataset_biom.iloc[i,19] = 3
  if dataset_biom.iloc[i,19] == 'Indifferently':
    dataset_biom.iloc[i,19] = 2
  if dataset_biom.iloc[i,19] == 'Won’t teach at all':
    dataset_biom.iloc[i,19] = 1

  if dataset_biom.iloc[i,20] == 'Always effective':
    dataset_biom.iloc[i,20] = 5
  if dataset_biom.iloc[i,20] == 'Sometimes effective':
    dataset_biom.iloc[i,20] = 4
  if dataset_biom.iloc[i,20] == 'Just satisfactorily':
    dataset_biom.iloc[i,20] = 3
  if dataset_biom.iloc[i,20] == 'Generally ineffective':
    dataset_biom.iloc[i,20] = 2
  if dataset_biom.iloc[i,20] == 'Very poor communication':
    dataset_biom.iloc[i,20] = 1

  if dataset_biom.iloc[i,21] == 'Excellent':
    dataset_biom.iloc[i,21] = 5
  if dataset_biom.iloc[i,21] == 'Very good':
    dataset_biom.iloc[i,21] = 4
  if dataset_biom.iloc[i,21] == 'Good':
    dataset_biom.iloc[i,21] = 3
  if dataset_biom.iloc[i,21] == 'Fair':
    dataset_biom.iloc[i,21] = 2
  if dataset_biom.iloc[i,21] == 'Poor':
    dataset_biom.iloc[i,21] = 1

  if dataset_biom.iloc[i,22] == 'Always fair':
    dataset_biom.iloc[i,22] = 5
  if dataset_biom.iloc[i,22] == 'Usually fair':
    dataset_biom.iloc[i,22] = 4
  if dataset_biom.iloc[i,22] == 'Sometimes unfair':
    dataset_biom.iloc[i,22] = 3
  if dataset_biom.iloc[i,22] == 'Usually unfair':
    dataset_biom.iloc[i,22] = 2
  if dataset_biom.iloc[i,22] == 'Unfair':
    dataset_biom.iloc[i,22] = 1

  if dataset_biom.iloc[i,23] == 'Every time':
    dataset_biom.iloc[i,23] = 5
  if dataset_biom.iloc[i,23] == 'Usually':
    dataset_biom.iloc[i,23] = 4
  if dataset_biom.iloc[i,23] == 'Occasionally/Sometimes':
    dataset_biom.iloc[i,23] = 3
  if dataset_biom.iloc[i,23] == 'Rarely':
    dataset_biom.iloc[i,23] = 2
  if dataset_biom.iloc[i,23] == 'Never':
    dataset_biom.iloc[i,23] = 1

  if dataset_biom.iloc[i,24] == 'Regularly':
    dataset_biom.iloc[i,24] = 5
  if dataset_biom.iloc[i,24] == 'Often':
    dataset_biom.iloc[i,24] = 4
  if dataset_biom.iloc[i,24] == 'Sometimes':
    dataset_biom.iloc[i,24] = 3
  if dataset_biom.iloc[i,24] == 'Rarely':
    dataset_biom.iloc[i,24] = 2
  if dataset_biom.iloc[i,24] == 'Never':
    dataset_biom.iloc[i,24] = 1

  if dataset_biom.iloc[i,25] == 'Significantly':
    dataset_biom.iloc[i,25] = 5
  if dataset_biom.iloc[i,25] == 'Very well':
    dataset_biom.iloc[i,25] = 4
  if dataset_biom.iloc[i,25] == 'Moderately':
    dataset_biom.iloc[i,25] = 3
  if dataset_biom.iloc[i,25] == 'Marginally':
    dataset_biom.iloc[i,25] = 2
  if dataset_biom.iloc[i,25] == 'Not at all':
    dataset_biom.iloc[i,25] = 1

  if dataset_biom.iloc[i,26] == 'Strongly agree':
    dataset_biom.iloc[i,26] = 5
  if dataset_biom.iloc[i,26] == 'Agree':
    dataset_biom.iloc[i,26] = 4
  if dataset_biom.iloc[i,26] == 'Neutral':
    dataset_biom.iloc[i,26] = 3
  if dataset_biom.iloc[i,26] == 'Disagree':
    dataset_biom.iloc[i,26] = 2
  if dataset_biom.iloc[i,26] == 'Strongly Disagree':
    dataset_biom.iloc[i,26] = 1

  if dataset_biom.iloc[i,27] == 'Every time':
    dataset_biom.iloc[i,27] = 5
  if dataset_biom.iloc[i,27] == 'Usually':
    dataset_biom.iloc[i,27] = 4
  if dataset_biom.iloc[i,27] == 'Occasionally/Sometimes':
    dataset_biom.iloc[i,27] = 3
  if dataset_biom.iloc[i,27] == 'Rarely':
    dataset_biom.iloc[i,27] = 2
  if dataset_biom.iloc[i,27] == 'Never':
    dataset_biom.iloc[i,27] = 1

  if dataset_biom.iloc[i,28] == 'Every time':
    dataset_biom.iloc[i,28] = 5
  if dataset_biom.iloc[i,28] == 'Usually':
    dataset_biom.iloc[i,28] = 4
  if dataset_biom.iloc[i,28] == 'Occasionally/Sometimes':
    dataset_biom.iloc[i,28] = 3
  if dataset_biom.iloc[i,28] == 'Rarely':
    dataset_biom.iloc[i,28] = 2
  if dataset_biom.iloc[i,28] == 'Never':
    dataset_biom.iloc[i,28] = 1

  if dataset_biom.iloc[i,29] == 'Every time':
    dataset_biom.iloc[i,29] = 5
  if dataset_biom.iloc[i,29] == 'Usually':
    dataset_biom.iloc[i,29] = 4
  if dataset_biom.iloc[i,29] == 'Occasionally/Sometimes':
    dataset_biom.iloc[i,29] = 3
  if dataset_biom.iloc[i,29] == 'Rarely':
    dataset_biom.iloc[i,29] = 2
  if dataset_biom.iloc[i,29] == 'Never':
    dataset_biom.iloc[i,29] = 1

  if dataset_biom.iloc[i,30] == 'Fully':
    dataset_biom.iloc[i,30] = 5
  if dataset_biom.iloc[i,30] == 'Reasonably':
    dataset_biom.iloc[i,30] = 4
  if dataset_biom.iloc[i,30] == 'Partially':
    dataset_biom.iloc[i,30] = 3
  if dataset_biom.iloc[i,30] == 'Slightly':
    dataset_biom.iloc[i,30] = 2
  if dataset_biom.iloc[i,30] == 'Unable to':
    dataset_biom.iloc[i,30] = 1

  if dataset_biom.iloc[i,31] == 'Every time':
    dataset_biom.iloc[i,31] = 5
  if dataset_biom.iloc[i,31] == 'Usually':
    dataset_biom.iloc[i,31] = 4
  if dataset_biom.iloc[i,31] == 'Occasionally/Sometimes':
    dataset_biom.iloc[i,31] = 3
  if dataset_biom.iloc[i,31] == 'Rarely':
    dataset_biom.iloc[i,31] = 2
  if dataset_biom.iloc[i,31] == 'Never':
    dataset_biom.iloc[i,31] = 1

  if dataset_biom.iloc[i,32] == 'Strongly agree':
    dataset_biom.iloc[i,32] = 5
  if dataset_biom.iloc[i,32] == 'Agree':
    dataset_biom.iloc[i,32] = 4
  if dataset_biom.iloc[i,32] == 'Neutral':
    dataset_biom.iloc[i,32] = 3
  if dataset_biom.iloc[i,32] == 'Disagree':
    dataset_biom.iloc[i,32] = 2
  if dataset_biom.iloc[i,32] == 'Strongly Disagree':
    dataset_biom.iloc[i,32] = 1

  if dataset_biom.iloc[i,33] == 'To a great extent':
    dataset_biom.iloc[i,33] = 5
  if dataset_biom.iloc[i,33] == 'Moderate':
    dataset_biom.iloc[i,33] = 4
  if dataset_biom.iloc[i,33] == 'Some what':
    dataset_biom.iloc[i,33] = 3
  if dataset_biom.iloc[i,33] == 'Very little':
    dataset_biom.iloc[i,33] = 2
  if dataset_biom.iloc[i,33] == 'Not at all':
    dataset_biom.iloc[i,33] = 1

  if dataset_biom.iloc[i,34] == 'Strongly agree':
    dataset_biom.iloc[i,34] = 5
  if dataset_biom.iloc[i,34] == 'Agree':
    dataset_biom.iloc[i,34] = 4
  if dataset_biom.iloc[i,34] == 'Neutral':
    dataset_biom.iloc[i,34] = 3
  if dataset_biom.iloc[i,34] == 'Disagree':
    dataset_biom.iloc[i,34] = 2
  if dataset_biom.iloc[i,34] == 'Strongly Disagree':
    dataset_biom.iloc[i,34] = 1

  if dataset_biom.iloc[i,35] == 'To a great extent':
    dataset_biom.iloc[i,35] = 5
  if dataset_biom.iloc[i,35] == 'Moderate':
    dataset_biom.iloc[i,35] = 4
  if dataset_biom.iloc[i,35] == 'Some what':
    dataset_biom.iloc[i,35] = 3
  if dataset_biom.iloc[i,35] == 'Very little':
    dataset_biom.iloc[i,35] = 2
  if dataset_biom.iloc[i,35] == 'Not at all':
    dataset_biom.iloc[i,35] = 1

  if dataset_biom.iloc[i,36] == 'Above 90%':
    dataset_biom.iloc[i,36] = 5
  if dataset_biom.iloc[i,36] == '70–89%':
    dataset_biom.iloc[i,36] = 4
  if dataset_biom.iloc[i,36] == '50–69%':
    dataset_biom.iloc[i,36] = 3
  if dataset_biom.iloc[i,36] == '30–49%':
    dataset_biom.iloc[i,36] = 2
  if dataset_biom.iloc[i,36] == 'Below 29%':
    dataset_biom.iloc[i,36] = 1

  if dataset_biom.iloc[i,37] == 'Strongly agree':
    dataset_biom.iloc[i,37] = 5
  if dataset_biom.iloc[i,37] == 'Agree':
    dataset_biom.iloc[i,37] = 4
  if dataset_biom.iloc[i,37] == 'Neutral':
    dataset_biom.iloc[i,37] = 3
  if dataset_biom.iloc[i,37] == 'Disagree':
    dataset_biom.iloc[i,37] = 2
  if dataset_biom.iloc[i,37] == 'Strongly Disagree':
    dataset_biom.iloc[i,37] = 1

for i in range(0,n_mms):
  if dataset_mms.iloc[i,18] == '85 to 100%':
    dataset_mms.iloc[i,18] = 5
  if dataset_mms.iloc[i,18] == '70 to 84%':
    dataset_mms.iloc[i,18] = 4
  if dataset_mms.iloc[i,18] == '55 to 69%':
    dataset_mms.iloc[i,18] = 3
  if dataset_mms.iloc[i,18] == '30 to 54%':
    dataset_mms.iloc[i,18] = 2
  if dataset_mms.iloc[i,18] == 'Below 30%':
    dataset_mms.iloc[i,18] = 1

  if dataset_mms.iloc[i,19] == 'Thoroughly':
    dataset_mms.iloc[i,19] = 5
  if dataset_mms.iloc[i,19] == 'Satisfactorily':
    dataset_mms.iloc[i,19] = 4
  if dataset_mms.iloc[i,19] == 'Poorly':
    dataset_mms.iloc[i,19] = 3
  if dataset_mms.iloc[i,19] == 'Indifferently':
    dataset_mms.iloc[i,19] = 2
  if dataset_mms.iloc[i,19] == 'Won’t teach at all':
    dataset_mms.iloc[i,19] = 1

  if dataset_mms.iloc[i,20] == 'Always effective':
    dataset_mms.iloc[i,20] = 5
  if dataset_mms.iloc[i,20] == 'Sometimes effective':
    dataset_mms.iloc[i,20] = 4
  if dataset_mms.iloc[i,20] == 'Just satisfactorily':
    dataset_mms.iloc[i,20] = 3
  if dataset_mms.iloc[i,20] == 'Generally ineffective':
    dataset_mms.iloc[i,20] = 2
  if dataset_mms.iloc[i,20] == 'Very poor communication':
    dataset_mms.iloc[i,20] = 1

  if dataset_mms.iloc[i,21] == 'Excellent':
    dataset_mms.iloc[i,21] = 5
  if dataset_mms.iloc[i,21] == 'Very good':
    dataset_mms.iloc[i,21] = 4
  if dataset_mms.iloc[i,21] == 'Good':
    dataset_mms.iloc[i,21] = 3
  if dataset_mms.iloc[i,21] == 'Fair':
    dataset_mms.iloc[i,21] = 2
  if dataset_mms.iloc[i,21] == 'Poor':
    dataset_mms.iloc[i,21] = 1

  if dataset_mms.iloc[i,22] == 'Always fair':
    dataset_mms.iloc[i,22] = 5
  if dataset_mms.iloc[i,22] == 'Usually fair':
    dataset_mms.iloc[i,22] = 4
  if dataset_mms.iloc[i,22] == 'Sometimes unfair':
    dataset_mms.iloc[i,22] = 3
  if dataset_mms.iloc[i,22] == 'Usually unfair':
    dataset_mms.iloc[i,22] = 2
  if dataset_mms.iloc[i,22] == 'Unfair':
    dataset_mms.iloc[i,22] = 1

  if dataset_mms.iloc[i,23] == 'Every time':
    dataset_mms.iloc[i,23] = 5
  if dataset_mms.iloc[i,23] == 'Usually':
    dataset_mms.iloc[i,23] = 4
  if dataset_mms.iloc[i,23] == 'Occasionally/Sometimes':
    dataset_mms.iloc[i,23] = 3
  if dataset_mms.iloc[i,23] == 'Rarely':
    dataset_mms.iloc[i,23] = 2
  if dataset_mms.iloc[i,23] == 'Never':
    dataset_mms.iloc[i,23] = 1

  if dataset_mms.iloc[i,24] == 'Regularly':
    dataset_mms.iloc[i,24] = 5
  if dataset_mms.iloc[i,24] == 'Often':
    dataset_mms.iloc[i,24] = 4
  if dataset_mms.iloc[i,24] == 'Sometimes':
    dataset_mms.iloc[i,24] = 3
  if dataset_mms.iloc[i,24] == 'Rarely':
    dataset_mms.iloc[i,24] = 2
  if dataset_mms.iloc[i,24] == 'Never':
    dataset_mms.iloc[i,24] = 1

  if dataset_mms.iloc[i,25] == 'Significantly':
    dataset_mms.iloc[i,25] = 5
  if dataset_mms.iloc[i,25] == 'Very well':
    dataset_mms.iloc[i,25] = 4
  if dataset_mms.iloc[i,25] == 'Moderately':
    dataset_mms.iloc[i,25] = 3
  if dataset_mms.iloc[i,25] == 'Marginally':
    dataset_mms.iloc[i,25] = 2
  if dataset_mms.iloc[i,25] == 'Not at all':
    dataset_mms.iloc[i,25] = 1

  if dataset_mms.iloc[i,26] == 'Strongly agree':
    dataset_mms.iloc[i,26] = 5
  if dataset_mms.iloc[i,26] == 'Agree':
    dataset_mms.iloc[i,26] = 4
  if dataset_mms.iloc[i,26] == 'Neutral':
    dataset_mms.iloc[i,26] = 3
  if dataset_mms.iloc[i,26] == 'Disagree':
    dataset_mms.iloc[i,26] = 2
  if dataset_mms.iloc[i,26] == 'Strongly Disagree':
    dataset_mms.iloc[i,26] = 1

  if dataset_mms.iloc[i,27] == 'Every time':
    dataset_mms.iloc[i,27] = 5
  if dataset_mms.iloc[i,27] == 'Usually':
    dataset_mms.iloc[i,27] = 4
  if dataset_mms.iloc[i,27] == 'Occasionally/Sometimes':
    dataset_mms.iloc[i,27] = 3
  if dataset_mms.iloc[i,27] == 'Rarely':
    dataset_mms.iloc[i,27] = 2
  if dataset_mms.iloc[i,27] == 'Never':
    dataset_mms.iloc[i,27] = 1

  if dataset_mms.iloc[i,28] == 'Every time':
    dataset_mms.iloc[i,28] = 5
  if dataset_mms.iloc[i,28] == 'Usually':
    dataset_mms.iloc[i,28] = 4
  if dataset_mms.iloc[i,28] == 'Occasionally/Sometimes':
    dataset_mms.iloc[i,28] = 3
  if dataset_mms.iloc[i,28] == 'Rarely':
    dataset_mms.iloc[i,28] = 2
  if dataset_mms.iloc[i,28] == 'Never':
    dataset_mms.iloc[i,28] = 1

  if dataset_mms.iloc[i,29] == 'Every time':
    dataset_mms.iloc[i,29] = 5
  if dataset_mms.iloc[i,29] == 'Usually':
    dataset_mms.iloc[i,29] = 4
  if dataset_mms.iloc[i,29] == 'Occasionally/Sometimes':
    dataset_mms.iloc[i,29] = 3
  if dataset_mms.iloc[i,29] == 'Rarely':
    dataset_mms.iloc[i,29] = 2
  if dataset_mms.iloc[i,29] == 'Never':
    dataset_mms.iloc[i,29] = 1

  if dataset_mms.iloc[i,30] == 'Fully':
    dataset_mms.iloc[i,30] = 5
  if dataset_mms.iloc[i,30] == 'Reasonably':
    dataset_mms.iloc[i,30] = 4
  if dataset_mms.iloc[i,30] == 'Partially':
    dataset_mms.iloc[i,30] = 3
  if dataset_mms.iloc[i,30] == 'Slightly':
    dataset_mms.iloc[i,30] = 2
  if dataset_mms.iloc[i,30] == 'Unable to':
    dataset_mms.iloc[i,30] = 1

  if dataset_mms.iloc[i,31] == 'Every time':
    dataset_mms.iloc[i,31] = 5
  if dataset_mms.iloc[i,31] == 'Usually':
    dataset_mms.iloc[i,31] = 4
  if dataset_mms.iloc[i,31] == 'Occasionally/Sometimes':
    dataset_mms.iloc[i,31] = 3
  if dataset_mms.iloc[i,31] == 'Rarely':
    dataset_mms.iloc[i,31] = 2
  if dataset_mms.iloc[i,31] == 'Never':
    dataset_mms.iloc[i,31] = 1

  if dataset_mms.iloc[i,32] == 'Strongly agree':
    dataset_mms.iloc[i,32] = 5
  if dataset_mms.iloc[i,32] == 'Agree':
    dataset_mms.iloc[i,32] = 4
  if dataset_mms.iloc[i,32] == 'Neutral':
    dataset_mms.iloc[i,32] = 3
  if dataset_mms.iloc[i,32] == 'Disagree':
    dataset_mms.iloc[i,32] = 2
  if dataset_mms.iloc[i,32] == 'Strongly Disagree':
    dataset_mms.iloc[i,32] = 1

  if dataset_mms.iloc[i,33] == 'To a great extent':
    dataset_mms.iloc[i,33] = 5
  if dataset_mms.iloc[i,33] == 'Moderate':
    dataset_mms.iloc[i,33] = 4
  if dataset_mms.iloc[i,33] == 'Some what':
    dataset_mms.iloc[i,33] = 3
  if dataset_mms.iloc[i,33] == 'Very little':
    dataset_mms.iloc[i,33] = 2
  if dataset_mms.iloc[i,33] == 'Not at all':
    dataset_mms.iloc[i,33] = 1

  if dataset_mms.iloc[i,34] == 'Strongly agree':
    dataset_mms.iloc[i,34] = 5
  if dataset_mms.iloc[i,34] == 'Agree':
    dataset_mms.iloc[i,34] = 4
  if dataset_mms.iloc[i,34] == 'Neutral':
    dataset_mms.iloc[i,34] = 3
  if dataset_mms.iloc[i,34] == 'Disagree':
    dataset_mms.iloc[i,34] = 2
  if dataset_mms.iloc[i,34] == 'Strongly Disagree':
    dataset_mms.iloc[i,34] = 1

  if dataset_mms.iloc[i,35] == 'To a great extent':
    dataset_mms.iloc[i,35] = 5
  if dataset_mms.iloc[i,35] == 'Moderate':
    dataset_mms.iloc[i,35] = 4
  if dataset_mms.iloc[i,35] == 'Some what':
    dataset_mms.iloc[i,35] = 3
  if dataset_mms.iloc[i,35] == 'Very little':
    dataset_mms.iloc[i,35] = 2
  if dataset_mms.iloc[i,35] == 'Not at all':
    dataset_mms.iloc[i,35] = 1

  if dataset_mms.iloc[i,36] == 'Above 90%':
    dataset_mms.iloc[i,36] = 5
  if dataset_mms.iloc[i,36] == '70–89%':
    dataset_mms.iloc[i,36] = 4
  if dataset_mms.iloc[i,36] == '50–69%':
    dataset_mms.iloc[i,36] = 3
  if dataset_mms.iloc[i,36] == '30–49%':
    dataset_mms.iloc[i,36] = 2
  if dataset_mms.iloc[i,36] == 'Below 29%':
    dataset_mms.iloc[i,36] = 1

  if dataset_mms.iloc[i,37] == 'Strongly agree':
    dataset_mms.iloc[i,37] = 5
  if dataset_mms.iloc[i,37] == 'Agree':
    dataset_mms.iloc[i,37] = 4
  if dataset_mms.iloc[i,37] == 'Neutral':
    dataset_mms.iloc[i,37] = 3
  if dataset_mms.iloc[i,37] == 'Disagree':
    dataset_mms.iloc[i,37] = 2
  if dataset_mms.iloc[i,37] == 'Strongly Disagree':
    dataset_mms.iloc[i,37] = 1

df200 = dataset2.iloc[:,18:38]

dff100 = list(df200.columns.values)
dff101 = pd.DataFrame(dff100)
dff101.columns = ['Questions']
dff101.insert(1,"Protractors",0)
dff101.insert(2,"Neutral",0)
dff101.insert(3,"Detractors",0)
dff101.insert(4,"NPS Score (CMPN)",0)
dff_cmpn = dff101

dataset_cmpn1 = dataset_cmpn.iloc[:,18:38]
dataset_inft1 = dataset_inft.iloc[:,18:38]
dataset_etrx1 = dataset_etrx.iloc[:,18:38]
dataset_extc1 = dataset_extc.iloc[:,18:38]
dataset_biom1 = dataset_biom.iloc[:,18:38]
dataset_mms1 = dataset_mms.iloc[:,18:38]

for i in range(0,20):
  for j in range(0,n_cmpn):
    if dataset_cmpn1.iloc[j,i] == 5:
      dff_cmpn.iloc[i,1] = dff_cmpn.iloc[i,1] + 1
    if dataset_cmpn1.iloc[j,i] == 4:
      dff_cmpn.iloc[i,2] = dff_cmpn.iloc[i,2] + 1
    if dataset_cmpn1.iloc[j,i] == 3:
      dff_cmpn.iloc[i,3] = dff_cmpn.iloc[i,3] + 1
    if dataset_cmpn1.iloc[j,i] == 2:
      dff_cmpn.iloc[i,3] = dff_cmpn.iloc[i,3] + 1
    if dataset_cmpn1.iloc[j,i] == 1:
      dff_cmpn.iloc[i,3] = dff_cmpn.iloc[i,3] + 1

dff_cmpn

df300 = dataset2.iloc[:,18:38]

dff400 = list(df300.columns.values)
dff401 = pd.DataFrame(dff400)
dff401.columns = ['Questions']
dff401.insert(1,"Protractors",0)
dff401.insert(2,"Neutral",0)
dff401.insert(3,"Detractors",0)
dff401.insert(4,"NPS Score (INFT)",0)
dff_inft = dff401

for i in range(0,20):
  for j in range(0,n_inft):
    if dataset_inft1.iloc[j,i] == 5:
      dff_inft.iloc[i,1] = dff_inft.iloc[i,1] + 1
    if dataset_inft1.iloc[j,i] == 4:
      dff_inft.iloc[i,2] = dff_inft.iloc[i,2] + 1
    if dataset_inft1.iloc[j,i] == 3:
      dff_inft.iloc[i,3] = dff_inft.iloc[i,3] + 1
    if dataset_inft1.iloc[j,i] == 2:
      dff_inft.iloc[i,3] = dff_inft.iloc[i,3] + 1
    if dataset_inft1.iloc[j,i] == 1:
      dff_inft.iloc[i,3] = dff_inft.iloc[i,3] + 1

dff_inft

df500 = dataset2.iloc[:,18:38]

dff600 = list(df500.columns.values)
dff601 = pd.DataFrame(dff600)
dff601.columns = ['Questions']
dff601.insert(1,"Protractors",0)
dff601.insert(2,"Neutral",0)
dff601.insert(3,"Detractors",0)
dff601.insert(4,"NPS Score (ETRX)",0)
dff_etrx = dff601

for i in range(0,20):
  for j in range(0,n_etrx):
    if dataset_etrx1.iloc[j,i] == 5:
      dff_etrx.iloc[i,1] = dff_etrx.iloc[i,1] + 1
    if dataset_etrx1.iloc[j,i] == 4:
      dff_etrx.iloc[i,2] = dff_etrx.iloc[i,2] + 1
    if dataset_etrx1.iloc[j,i] == 3:
      dff_etrx.iloc[i,3] = dff_etrx.iloc[i,3] + 1
    if dataset_etrx1.iloc[j,i] == 2:
      dff_etrx.iloc[i,3] = dff_etrx.iloc[i,3] + 1
    if dataset_etrx1.iloc[j,i] == 1:
      dff_etrx.iloc[i,3] = dff_etrx.iloc[i,3] + 1

dff_etrx

df700 = dataset2.iloc[:,18:38]

dff800 = list(df700.columns.values)
dff801 = pd.DataFrame(dff800)
dff801.columns = ['Questions']
dff801.insert(1,"Protractors",0)
dff801.insert(2,"Neutral",0)
dff801.insert(3,"Detractors",0)
dff801.insert(4,"NPS Score (EXTC)",0)
dff_extc = dff801

for i in range(0,20):
  for j in range(0,n_extc):
    if dataset_extc1.iloc[j,i] == 5:
      dff_extc.iloc[i,1] = dff_extc.iloc[i,1] + 1
    if dataset_extc1.iloc[j,i] == 4:
      dff_extc.iloc[i,2] = dff_extc.iloc[i,2] + 1
    if dataset_extc1.iloc[j,i] == 3:
      dff_extc.iloc[i,3] = dff_extc.iloc[i,3] + 1
    if dataset_extc1.iloc[j,i] == 2:
      dff_extc.iloc[i,3] = dff_extc.iloc[i,3] + 1
    if dataset_extc1.iloc[j,i] == 1:
      dff_extc.iloc[i,3] = dff_extc.iloc[i,3] + 1

dff_extc

df900 = dataset2.iloc[:,18:38]

dff1000 = list(df900.columns.values)
dff1001 = pd.DataFrame(dff1000)
dff1001.columns = ['Questions']
dff1001.insert(1,"Protractors",0)
dff1001.insert(2,"Neutral",0)
dff1001.insert(3,"Detractors",0)
dff1001.insert(4,"NPS Score (BIOMED)",0)
dff_biom = dff1001

for i in range(0,20):
  for j in range(0,n_biom):
    if dataset_biom1.iloc[j,i] == 5:
      dff_biom.iloc[i,1] = dff_biom.iloc[i,1] + 1
    if dataset_biom1.iloc[j,i] == 4:
      dff_biom.iloc[i,2] = dff_biom.iloc[i,2] + 1
    if dataset_biom1.iloc[j,i] == 3:
      dff_biom.iloc[i,3] = dff_biom.iloc[i,3] + 1
    if dataset_biom1.iloc[j,i] == 2:
      dff_biom.iloc[i,3] = dff_biom.iloc[i,3] + 1
    if dataset_biom1.iloc[j,i] == 1:
      dff_biom.iloc[i,3] = dff_biom.iloc[i,3] + 1

dff_biom

df1100 = dataset2.iloc[:,18:38]

dff1200 = list(df1100.columns.values)
dff1201 = pd.DataFrame(dff1200)
dff1201.columns = ['Questions']
dff1201.insert(1,"Protractors",0)
dff1201.insert(2,"Neutral",0)
dff1201.insert(3,"Detractors",0)
dff1201.insert(4,"NPS Score (MMS)",0)
dff_mms = dff1201

for i in range(0,20):
  for j in range(0,n_mms):
    if dataset_mms1.iloc[j,i] == 5:
      dff_mms.iloc[i,1] = dff_mms.iloc[i,1] + 1
    if dataset_mms1.iloc[j,i] == 4:
      dff_mms.iloc[i,2] = dff_mms.iloc[i,2] + 1
    if dataset_mms1.iloc[j,i] == 3:
      dff_mms.iloc[i,3] = dff_mms.iloc[i,3] + 1
    if dataset_mms1.iloc[j,i] == 2:
      dff_mms.iloc[i,3] = dff_mms.iloc[i,3] + 1
    if dataset_mms1.iloc[j,i] == 1:
      dff_mms.iloc[i,3] = dff_mms.iloc[i,3] + 1

dff_mms

dff_cmpn['Protractors'] = dff_cmpn['Protractors']*100/n_cmpn
dff_cmpn['Neutral'] = dff_cmpn['Neutral']*100/n_cmpn
dff_cmpn['Detractors'] = dff_cmpn['Detractors']*100/n_cmpn
dff_cmpn['NPS Score (CMPN)'] = dff_cmpn['Protractors'] - dff_cmpn['Detractors']

dff_inft['Protractors'] = dff_inft['Protractors']*100/n_inft
dff_inft['Neutral'] = dff_inft['Neutral']*100/n_inft
dff_inft['Detractors'] = dff_inft['Detractors']*100/n_inft
dff_inft['NPS Score (INFT)'] = dff_inft['Protractors'] - dff_inft['Detractors']

dff_etrx['Protractors'] = dff_etrx['Protractors']*100/n_etrx
dff_etrx['Neutral'] = dff_etrx['Neutral']*100/n_etrx
dff_etrx['Detractors'] = dff_etrx['Detractors']*100/n_etrx
dff_etrx['NPS Score (ETRX)'] = dff_etrx['Protractors'] - dff_etrx['Detractors']

dff_extc['Protractors'] = dff_extc['Protractors']*100/n_extc
dff_extc['Neutral'] = dff_extc['Neutral']*100/n_extc
dff_extc['Detractors'] = dff_extc['Detractors']*100/n_extc
dff_extc['NPS Score (EXTC)'] = dff_extc['Protractors'] - dff_extc['Detractors']

dff_biom['Protractors'] = dff_biom['Protractors']*100/n_biom
dff_biom['Neutral'] = dff_biom['Neutral']*100/n_biom
dff_biom['Detractors'] = dff_biom['Detractors']*100/n_biom
dff_biom['NPS Score (BIOMED)'] = dff_biom['Protractors'] - dff_biom['Detractors']

dff_mms['Protractors'] = dff_mms['Protractors']*100/n_mms
dff_mms['Neutral'] = dff_mms['Neutral']*100/n_mms
dff_mms['Detractors'] = dff_mms['Detractors']*100/n_mms
dff_mms['NPS Score (MMS)'] = dff_mms['Protractors'] - dff_mms['Detractors']

dff_cmpn

dff_inft

dff_etrx

dff_extc

dff_biom

dff_mms

nps_final_df = dff_cmpn[['Questions','NPS Score (CMPN)']]

nps_final_df = pd.concat([nps_final_df, dff_inft.iloc[:,-1], dff_etrx.iloc[:,-1], dff_extc.iloc[:,-1],
                         dff_biom.iloc[:,-1], dff_mms.iloc[:,-1]], axis=1)

nps_final_df.to_csv('NPS SCORE.csv')

"""## Suggestions - NLP

### CMPN
"""

dataset_cmpn_reviews = dataset_cmpn.iloc[:,-1]
dataset_cmpn_reviews = pd.DataFrame(dataset_cmpn_reviews)

dataset_cmpn_reviews['index'] = range(1, 1+len(dataset_cmpn_reviews))
dataset_cmpn_reviews.set_index('index', inplace=True)
dataset_cmpn_reviews.columns = ['Suggestions']
dataset_cmpn_reviews

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus_cmpn = []
for i in range(1, len(dataset_cmpn_reviews)+1):
  sugg = re.sub("[^a-zA-Z]", " ", dataset_cmpn_reviews['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [wordnet.lemmatize(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus_cmpn.append(sugg)

corpus_cmpn

x_cmpn = vectorizer.transform(dataset_cmpn_reviews['Suggestions'])
ypred_cmpn = model.predict(x_cmpn.toarray())
dataset_cmpn_reviews.insert(1, "Sentiment",ypred_cmpn)
for i in range (0,n_cmpn):
  if dataset_cmpn_reviews.iloc[i,1]>0.6:
    dataset_cmpn_reviews.iloc[i,1]=1
  if dataset_cmpn_reviews.iloc[i,1]<0.4:
    dataset_cmpn_reviews.iloc[i,1]=0

dataset_cmpn_reviews

list_cmpn_pos = dataset_cmpn_reviews[dataset_cmpn_reviews['Sentiment']==1]['Suggestions'].tolist()
list_cmpn_neg = dataset_cmpn_reviews[dataset_cmpn_reviews['Sentiment']==0]['Suggestions'].tolist()

list_cmpn_pos

import nltk
nltk.download('punkt')

import nltk
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams


from collections import Counter

non_speaker = re.compile('[A-Za-z]+: (.*)')

def untokenize(ngram):
    tokens = list(ngram)
    return "".join([" "+i if not i.startswith("'") and
                    i not in string.punctuation and
                    i != "n't"
                    else i for i in tokens]).strip()

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            phrase_counter[untokenize(phrase)] += 1

phrase_counter_pos = Counter()
phrase_counter_neg = Counter()

sentencesfile = dataset

for i in range(0, len(list_cmpn_pos)):
    extract_phrases(list_cmpn_pos[i], phrase_counter_pos, 2)

for i in range(0, len(list_cmpn_neg)):
    extract_phrases(list_cmpn_neg[i], phrase_counter_neg, 2)

cmpn_pos = phrase_counter_pos.most_common(10)
cmpn_neg = phrase_counter_neg.most_common(10)
for k,v in cmpn_pos:
    print (k,"-",v,"times - Positive")
for p,q in cmpn_neg:
    print(p,"-",q,"times - Negative")

cmpn_final_pos = pd.DataFrame(cmpn_pos)
cmpn_final_neg = pd.DataFrame(cmpn_neg)
try:
    cmpn_final_pos.columns = ['Reviews (Positive)','Occurrence']
except:
    cmpn_final_neg.columns = ['Reviews (Negative)','Occurrence']

try:
    cmpn_final_neg.columns = ['Reviews (Negative)','Occurrence']
except:
    cmpn_final_pos.columns = ['Reviews (Positive)','Occurrence']

cmpn_final = pd.concat([cmpn_final_pos, cmpn_final_neg], axis=1)

cmpn_final.to_csv('CMPN_FINAL_REVIEWS.csv')

"""### INFT"""

dataset_inft_reviews = dataset_inft.iloc[:,-1]
dataset_inft_reviews = pd.DataFrame(dataset_inft_reviews)
dataset_inft_reviews['index'] = range(1, 1+len(dataset_inft_reviews))
dataset_inft_reviews.set_index('index', inplace=True)
dataset_inft_reviews.columns = ['Suggestions']
dataset_inft_reviews

from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus_inft = []
for i in range(1, len(dataset_inft_reviews)+1):
  sugg = re.sub("[^a-zA-Z]", " ", dataset_inft_reviews['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [wordnet.lemmatize(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus_inft.append(sugg)

x_inft = vectorizer.transform(dataset_inft_reviews['Suggestions'])
ypred_inft = model.predict(x_inft.toarray())
dataset_inft_reviews.insert(1, "Sentiment",ypred_inft)
for i in range (0,n_inft):
  if dataset_inft_reviews.iloc[i,1]>0.6:
    dataset_inft_reviews.iloc[i,1]=1
  if dataset_inft_reviews.iloc[i,1]<0.4:
    dataset_inft_reviews.iloc[i,1]=0
dataset_inft_reviews

list_inft_pos = dataset_inft_reviews[dataset_inft_reviews['Sentiment']==1]['Suggestions'].tolist()
list_inft_neg = dataset_inft_reviews[dataset_inft_reviews['Sentiment']==0]['Suggestions'].tolist()

import nltk
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams


from collections import Counter

non_speaker = re.compile('[A-Za-z]+: (.*)')

def untokenize(ngram):
    tokens = list(ngram)
    return "".join([" "+i if not i.startswith("'") and
                    i not in string.punctuation and
                    i != "n't"
                    else i for i in tokens]).strip()

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            phrase_counter[untokenize(phrase)] += 1

phrase_counter_pos = Counter()
phrase_counter_neg = Counter()

sentencesfile = dataset

for i in range(0, len(list_inft_pos)):
    extract_phrases(list_inft_pos[i], phrase_counter_pos, 2)

for i in range(0, len(list_inft_neg)):
    extract_phrases(list_inft_neg[i], phrase_counter_neg, 2)

inft_pos = phrase_counter_pos.most_common(10)
inft_neg = phrase_counter_neg.most_common(10)
for k,v in inft_pos:
    print (k,"-",v,"times - Positive")
for p,q in inft_neg:
    print(p,"-",q,"times - Negative")

inft_final_pos = pd.DataFrame(inft_pos)
inft_final_neg = pd.DataFrame(inft_neg)
try:
    inft_final_pos.columns = ['Reviews (Positive)','Occurrence']
except:
    inft_final_neg.columns = ['Reviews (Negative)','Occurrence']

try:
    inft_final_neg.columns = ['Reviews (Negative)','Occurrence']
except:
    inft_final_pos.columns = ['Reviews (Positive)','Occurrence']

inft_final = pd.concat([inft_final_pos, inft_final_neg], axis=1)
inft_final.to_csv('INFT_FINAL_REVIEWS.csv')

"""### EXTC"""

dataset_extc_reviews = dataset_extc.iloc[:,-1]
dataset_extc_reviews = pd.DataFrame(dataset_extc_reviews)
dataset_extc_reviews['index'] = range(1, 1+len(dataset_extc_reviews))
dataset_extc_reviews.set_index('index', inplace=True)
dataset_extc_reviews.columns = ['Suggestions']
dataset_extc_reviews

from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus_extc = []
for i in range(1, len(dataset_extc_reviews)+1):
  sugg = re.sub("[^a-zA-Z]", " ", dataset_extc_reviews['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [wordnet.lemmatize(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus_extc.append(sugg)

x_extc = vectorizer.transform(dataset_extc_reviews['Suggestions'])
ypred_extc = model.predict(x_extc.toarray())
dataset_extc_reviews.insert(1, "Sentiment",ypred_extc)
for i in range (0,n_extc):
  if dataset_extc_reviews.iloc[i,1]>0.6:
    dataset_extc_reviews.iloc[i,1]=1
  if dataset_extc_reviews.iloc[i,1]<0.4:
    dataset_extc_reviews.iloc[i,1]=0
dataset_extc_reviews

list_extc_pos = dataset_extc_reviews[dataset_extc_reviews['Sentiment']==1]['Suggestions'].tolist()
list_extc_neg = dataset_extc_reviews[dataset_extc_reviews['Sentiment']==0]['Suggestions'].tolist()

import nltk
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams


from collections import Counter

non_speaker = re.compile('[A-Za-z]+: (.*)')

def untokenize(ngram):
    tokens = list(ngram)
    return "".join([" "+i if not i.startswith("'") and
                    i not in string.punctuation and
                    i != "n't"
                    else i for i in tokens]).strip()

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            phrase_counter[untokenize(phrase)] += 1

phrase_counter_pos = Counter()
phrase_counter_neg = Counter()

sentencesfile = dataset

for i in range(0, len(list_extc_pos)):
    extract_phrases(list_extc_pos[i], phrase_counter_pos, 2)

for i in range(0, len(list_extc_neg)):
    extract_phrases(list_extc_neg[i], phrase_counter_neg, 2)

extc_pos = phrase_counter_pos.most_common(10)
extc_neg = phrase_counter_neg.most_common(10)
for k,v in extc_pos:
    print (k,"-",v,"times - Positive")
for p,q in extc_neg:
    print(p,"-",q,"times - Negative")

extc_final_pos = pd.DataFrame(extc_pos)
extc_final_neg = pd.DataFrame(extc_neg)
try:
    extc_final_pos.columns = ['Reviews (Positive)','Occurrence']
except:
    extc_final_neg.columns = ['Reviews (Negative)','Occurrence']

try:
    extc_final_neg.columns = ['Reviews (Negative)','Occurrence']
except:
    extc_final_pos.columns = ['Reviews (Positive)','Occurrence']

extc_final = pd.concat([extc_final_pos, extc_final_neg], axis=1)
extc_final.to_csv('EXTC_FINAL_REVIEWS.csv')

"""### ETRX"""

dataset_etrx_reviews = dataset_etrx.iloc[:,-1]
dataset_etrx_reviews = pd.DataFrame(dataset_etrx_reviews)
dataset_etrx_reviews['index'] = range(1, 1+len(dataset_etrx_reviews))
dataset_etrx_reviews.set_index('index', inplace=True)
dataset_etrx_reviews.columns = ['Suggestions']
dataset_etrx_reviews

from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus_etrx = []
for i in range(1, len(dataset_etrx_reviews)+1):
  sugg = re.sub("[^a-zA-Z]", " ", dataset_etrx_reviews['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [wordnet.lemmatize(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus_etrx.append(sugg)

x_etrx = vectorizer.transform(dataset_etrx_reviews['Suggestions'])
ypred_etrx = model.predict(x_etrx.toarray())
dataset_etrx_reviews.insert(1, "Sentiment",ypred_etrx)
for i in range (0,n_etrx):
  if dataset_etrx_reviews.iloc[i,1]>0.6:
    dataset_etrx_reviews.iloc[i,1]=1
  if dataset_etrx_reviews.iloc[i,1]<0.4:
    dataset_etrx_reviews.iloc[i,1]=0
dataset_etrx_reviews

list_etrx_pos = dataset_etrx_reviews[dataset_etrx_reviews['Sentiment']==1]['Suggestions'].tolist()
list_etrx_neg = dataset_etrx_reviews[dataset_etrx_reviews['Sentiment']==0]['Suggestions'].tolist()

import nltk
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams


from collections import Counter

non_speaker = re.compile('[A-Za-z]+: (.*)')

def untokenize(ngram):
    tokens = list(ngram)
    return "".join([" "+i if not i.startswith("'") and
                    i not in string.punctuation and
                    i != "n't"
                    else i for i in tokens]).strip()

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            phrase_counter[untokenize(phrase)] += 1

phrase_counter_pos = Counter()
phrase_counter_neg = Counter()

sentencesfile = dataset

for i in range(0, len(list_etrx_pos)):
    extract_phrases(list_etrx_pos[i], phrase_counter_pos, 2)

for i in range(0, len(list_etrx_neg)):
    extract_phrases(list_etrx_neg[i], phrase_counter_neg, 2)

etrx_pos = phrase_counter_pos.most_common(10)
etrx_neg = phrase_counter_neg.most_common(10)
for k,v in etrx_pos:
    print (k,"-",v,"times - Positive")
for p,q in etrx_neg:
    print(p,"-",q,"times - Negative")

etrx_final_pos = pd.DataFrame(etrx_pos)
etrx_final_neg = pd.DataFrame(etrx_neg)
try:
    etrx_final_pos.columns = ['Reviews (Positive)','Occurrence']
except:
    etrx_final_neg.columns = ['Reviews (Negative)','Occurrence']

try:
    etrx_final_neg.columns = ['Reviews (Negative)','Occurrence']
except:
    etrx_final_pos.columns = ['Reviews (Positive)','Occurrence']

etrx_final = pd.concat([etrx_final_pos, etrx_final_neg], axis=1)
etrx_final.to_csv('ETRX_FINAL_REVIEWS.csv')

"""### BIOMED"""

dataset_biom_reviews = dataset_biom.iloc[:,-1]
dataset_biom_reviews = pd.DataFrame(dataset_biom_reviews)
dataset_biom_reviews['index'] = range(1, 1+len(dataset_biom_reviews))
dataset_biom_reviews.set_index('index', inplace=True)
dataset_biom_reviews.columns = ['Suggestions']
dataset_biom_reviews

from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus_biom = []
for i in range(1, len(dataset_biom_reviews)+1):
  sugg = re.sub("[^a-zA-Z]", " ", dataset_biom_reviews['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [wordnet.lemmatize(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus_biom.append(sugg)

x_biom = vectorizer.transform(dataset_biom_reviews['Suggestions'])
ypred_biom = model.predict(x_biom.toarray())
dataset_biom_reviews.insert(1, "Sentiment",ypred_biom)
for i in range (0,n_biom):
  if dataset_biom_reviews.iloc[i,1]>0.6:
    dataset_biom_reviews.iloc[i,1]=1
  if dataset_biom_reviews.iloc[i,1]<0.4:
    dataset_biom_reviews.iloc[i,1]=0
dataset_biom_reviews

list_biom_pos = dataset_biom_reviews[dataset_biom_reviews['Sentiment']==1]['Suggestions'].tolist()
list_biom_neg = dataset_biom_reviews[dataset_biom_reviews['Sentiment']==0]['Suggestions'].tolist()

import nltk
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams


from collections import Counter

non_speaker = re.compile('[A-Za-z]+: (.*)')

def untokenize(ngram):
    tokens = list(ngram)
    return "".join([" "+i if not i.startswith("'") and
                    i not in string.punctuation and
                    i != "n't"
                    else i for i in tokens]).strip()

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            phrase_counter[untokenize(phrase)] += 1

phrase_counter_pos = Counter()
phrase_counter_neg = Counter()

sentencesfile = dataset

for i in range(0, len(list_biom_pos)):
    extract_phrases(list_biom_pos[i], phrase_counter_pos, 2)

for i in range(0, len(list_biom_neg)):
    extract_phrases(list_biom_neg[i], phrase_counter_neg, 2)

biom_pos = phrase_counter_pos.most_common(10)
biom_neg = phrase_counter_neg.most_common(10)
for k,v in biom_pos:
    print (k,"-",v,"times - Positive")
for p,q in biom_neg:
    print(p,"-",q,"times - Negative")

biom_final_pos = pd.DataFrame(biom_pos)
biom_final_neg = pd.DataFrame(biom_neg)
try:
    biom_final_pos.columns = ['Reviews (Positive)','Occurrence']
except:
    biom_final_neg.columns = ['Reviews (Negative)','Occurrence']

try:
    biom_final_neg.columns = ['Reviews (Negative)','Occurrence']
except:
    biom_final_pos.columns = ['Reviews (Positive)','Occurrence']

biom_final = pd.concat([biom_final_pos, biom_final_neg], axis=1)
biom_final.to_csv('BIOM_FINAL_REVIEWS.csv')

"""### MMS"""

dataset_mms_reviews = dataset_mms.iloc[:,-1]
dataset_mms_reviews = pd.DataFrame(dataset_mms_reviews)
dataset_mms_reviews['index'] = range(1, 1+len(dataset_mms_reviews))
dataset_mms_reviews.set_index('index', inplace=True)
dataset_mms_reviews.columns = ['Suggestions']
dataset_mms_reviews

from nltk.stem import WordNetLemmatizer
wordnet = WordNetLemmatizer()

corpus_mms = []
for i in range(1, len(dataset_mms_reviews)+1):
  sugg = re.sub("[^a-zA-Z]", " ", dataset_mms_reviews['Suggestions'][i])
  sugg = sugg.lower()
  sugg = sugg.split()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  sugg = [wordnet.lemmatize(word) for word in sugg if not word in set(all_stopwords)]
  sugg = ' '.join(sugg)
  corpus_mms.append(sugg)

x_mms = vectorizer.transform(dataset_mms_reviews['Suggestions'])
ypred_mms = model.predict(x_mms.toarray())
dataset_mms_reviews.insert(1, "Sentiment",ypred_mms)
for i in range (0,n_mms):
  if dataset_mms_reviews.iloc[i,1]>0.6:
    dataset_mms_reviews.iloc[i,1]=1
  if dataset_mms_reviews.iloc[i,1]<0.4:
    dataset_mms_reviews.iloc[i,1]=0
dataset_mms_reviews

list_mms_pos = dataset_mms_reviews[dataset_mms_reviews['Sentiment']==1]['Suggestions'].tolist()
list_mms_neg = dataset_mms_reviews[dataset_mms_reviews['Sentiment']==0]['Suggestions'].tolist()

import nltk
import csv
import string
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams


from collections import Counter

non_speaker = re.compile('[A-Za-z]+: (.*)')

def untokenize(ngram):
    tokens = list(ngram)
    return "".join([" "+i if not i.startswith("'") and
                    i not in string.punctuation and
                    i != "n't"
                    else i for i in tokens]).strip()

def extract_phrases(text, phrase_counter, length):
    for sent in nltk.sent_tokenize(text):
        strip_speaker = non_speaker.match(sent)
        if strip_speaker is not None:
            sent = strip_speaker.group(1)
        words = nltk.word_tokenize(sent)
        for phrase in ngrams(words, length):
            phrase_counter[untokenize(phrase)] += 1

phrase_counter_pos = Counter()
phrase_counter_neg = Counter()

sentencesfile = dataset

for i in range(0, len(list_mms_pos)):
    extract_phrases(list_mms_pos[i], phrase_counter_pos, 2)

for i in range(0, len(list_mms_neg)):
    extract_phrases(list_mms_neg[i], phrase_counter_neg, 2)

mms_pos = phrase_counter_pos.most_common(10)
mms_neg = phrase_counter_neg.most_common(10)
for k,v in mms_pos:
    print (k,"-",v,"times - Positive")
for p,q in mms_neg:
    print(p,"-",q,"times - Negative")

mms_final_pos = pd.DataFrame(mms_pos)
mms_final_neg = pd.DataFrame(mms_neg)
try:
    mms_final_pos.columns = ['Reviews (Positive)','Occurrence']
except:
    mms_final_neg.columns = ['Reviews (Negative)','Occurrence']

try:
    mms_final_neg.columns = ['Reviews (Negative)','Occurrence']
except:
    mms_final_pos.columns = ['Reviews (Positive)','Occurrence']

mms_final = pd.concat([mms_final_pos, mms_final_neg], axis=1)
mms_final.to_csv('MMS_FINAL_REVIEWS.csv')

dataset_cmpn_reviews

"""## SUGGESTIONS - DEPARTMENT WISE SCORE"""

# CMPN SUGGESTIONS

print("***COMPUTER DEPARTMENT SUGGESTIONS STATS***")
positive = 0
negative = 0
for i in range(1,len(dataset_cmpn_reviews)+1):
  if dataset_cmpn_reviews['Sentiment'][i] == 0:
    negative = negative + 1
  if dataset_cmpn_reviews['Sentiment'][i] == 1:
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/len(dataset_cmpn_reviews))
print("Percentage negative responses/suggestions = ",negative*100/len(dataset_cmpn_reviews))


# INFT SUGGESTIONS

print("\n***INFORMARION TECHNOLOGY DEPARTMENT SUGGESTIONS STATS***")
positive = 0
negative = 0
for i in range(1,len(dataset_inft_reviews)+1):
  if dataset_inft_reviews['Sentiment'][i] == 0:
    negative = negative + 1
  if dataset_inft_reviews['Sentiment'][i] == 1:
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/len(dataset_inft_reviews))
print("Percentage negative responses/suggestions = ",negative*100/len(dataset_inft_reviews))


# EXTC SUGGESTIONS

print("\n***EXTC DEPARTMENT SUGGESTIONS STATS***")
positive = 0
negative = 0
for i in range(1,len(dataset_extc_reviews)+1):
  if dataset_extc_reviews['Sentiment'][i] == 0:
    negative = negative + 1
  if dataset_extc_reviews['Sentiment'][i] == 1:
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/len(dataset_extc_reviews))
print("Percentage negative responses/suggestions = ",negative*100/len(dataset_extc_reviews))


# ETRX SUGGESTIONS

print("\n***ETRX DEPARTMENT SUGGESTIONS STATS***")
positive = 0
negative = 0
for i in range(1,len(dataset_etrx_reviews)+1):
  if dataset_etrx_reviews['Sentiment'][i] == 0:
    negative = negative + 1
  if dataset_etrx_reviews['Sentiment'][i] == 1:
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/len(dataset_etrx_reviews))
print("Percentage negative responses/suggestions = ",negative*100/len(dataset_etrx_reviews))


# BIOMED SUGGESTIONS

print("\n***BIOMEDICAL DEPARTMENT SUGGESTIONS STATS***")
positive = 0
negative = 0
for i in range(1,len(dataset_biom_reviews)+1):
  if dataset_biom_reviews['Sentiment'][i] == 0:
    negative = negative + 1
  if dataset_biom_reviews['Sentiment'][i] == 1:
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/len(dataset_biom_reviews))
print("Percentage negative responses/suggestions = ",negative*100/len(dataset_biom_reviews))



# MMS SUGGESTIONS

print("\n***MMS DEPARTMENT SUGGESTIONS STATS***")
positive = 0
negative = 0
for i in range(1,len(dataset_mms_reviews)+1):
  if dataset_mms_reviews['Sentiment'][i] == 0:
    negative = negative + 1
  if dataset_mms_reviews['Sentiment'][i] == 1:
    positive = positive + 1

print("Total positive responses/suggestions = ",positive)
print("Total negative responses/suggestions = ",negative)
print("Percentage positive responses/suggestions = ",positive*100/len(dataset_mms_reviews))
print("Percentage negative responses/suggestions = ",negative*100/len(dataset_mms_reviews))
