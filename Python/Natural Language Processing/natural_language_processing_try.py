# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 
# quoting - to ignore the double quotes

# Cleaning texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Including ^ indicates that the characters following it are not supposed to be removed
    # The characters removed other than the ones in the first argument are replaced by the value in the sencond argument that is space
    # The third argumnet indicates where the operations should occur
    review = review.lower()
    review = review.split() # to convert the string into list of words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Set function is used for faster execuion in case of longer texts with many words
    # combining the list of words to a string again
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#---------------------------------------------Naive Bayes------------------------------------------------#
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
Precision = 100 * TP / (TP + FP)
Recall = 100 * TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

#------------------------------------------Decision Tree----------------------------------------------#

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
# With entropy
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
Precision = 100 * TP / (TP + FP)
Recall = 100 * TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

#With Gini Index
classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
Precision = 100 * TP / (TP + FP)
Recall = 100 * TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)

#--------------------------------------------Random Forest-----------------------------------------------#

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]
Accuracy = 100 * (TP + TN) / (TP + TN + FP + FN)
Precision = 100 * TP / (TP + FP)
Recall = 100 * TP / (TP + FN)
F1_Score = 2 * Precision * Recall / (Precision + Recall)
