import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(3)

df1= pd.read_excel('FakeNewsDataset.xlsx', header=None)


df1=df1.dropna()


df1 = df1.sample(frac=1, random_state=42)


Y = df1.iloc[:,1].values


doc_complete=df1[0]
doc_clean = [doc.split() for doc in doc_complete]


list1=[]
for doc in doc_clean:
    list1=list1+doc
words = list(set(list1))


#words.append("ENDPAD")
word2idx = {w: i for i, w in enumerate(words)}

X=[]
for doc in doc_clean:
    temp=[]
    for j in doc:
        try:
            temp.append(word2idx[j])
        except KeyError:
            temp.append(word2idx["UNKNOWN_TOKEN"])
        
    X.append(temp)

maxlen = max([len(s) for s in X])

value=len(word2idx)



# Need not to convert it to Np.array
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(sequences=X, padding="post",value=value)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
 
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
                         
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)



from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

cm=confusion_matrix(y_test, y_pred)
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='binary')

#Accuracy 
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')




