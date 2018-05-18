from __future__ import print_function, division
from future.utils import iteritems
from builtins import range
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# importing dataset
df = pd.read_csv('spam.csv', encoding = 'ISO-8859-1') 
# This encoding is used since the csv contains some invalid characters and using default encoding can throw errors

# Drop unnecessary columns
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)

# Renaming the columns
df.columns = ['labels', 'data']

# Creating binary labels
df['b_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['b_labels'].as_matrix()

# Feature calculation using TF-IDF
tfidf = TfidfVectorizer(decode_error = 'ignore')
# if any invalid errors are found ignore them
X = tfidf.fit_transform(df['data'])

# Splitting data into train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.33)

# Creating the model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Train Score: ", model.score(Xtrain, Ytrain))
print("Test Score: ", model.score(Xtest, Ytest))


# Feature calculation using Count Vectorizer
CV = CountVectorizer(decode_error = 'ignore')
# if any invalid errors are found ignore them
X = CV.fit_transform(df['data'])

# Splitting data into train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.33)

# Creating the model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print("Train Score: ", model.score(Xtrain, Ytrain))
print("Test Score: ", model.score(Xtest, Ytest))

# Visualize the data
def visualize (label):
    words = ''
    for msg in df[df['labels']==label]['data']:
        msg = msg.lower()
        words += msg + ''
    wordcloud = WordCloud(width = 600, height = 400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
visualize('spam')
visualize('ham')

# Predicting the values and adding them into a new column
df['predictions'] = model.predict(X)

# Misclassifications
# Should be spam
spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']
for msg in spam:
    print(msg)
    
# Should not be spam
notspam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']
for msg1 in notspam:
    print(msg1)