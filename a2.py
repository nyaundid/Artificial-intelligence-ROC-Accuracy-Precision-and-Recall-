
# coding: utf-8

# In[26]:


import tensorflow as tf
import numpy as np
import os 
import pandas as pd 
import keras
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer
from keras import optimizers
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam    
from keras.metrics import categorical_crossentropy
from keras import backend as k


# In[27]:


dataset = "Documents\\CellDNA1.csv"
DNA = pd.read_csv( dataset,header=None, )
dataset = np.genfromtxt("Documents\\CellDNA1.csv", delimiter = ',')


# In[28]:


#full data set

DNA


# In[29]:


from scipy.stats import zscore
nz = DNA.apply(zscore)


# In[30]:


nz


# In[31]:


nz2 = DNA.loc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]


# In[32]:


#only colums 0-12 and z score

nz3 = nz2.apply(zscore)


# In[33]:


nz2


# In[34]:


nz3


# In[35]:


y_train= DNA.loc[:, [13]]


# In[36]:


y_train


# In[37]:


y1 = DNA.loc[:, [13]]


# In[38]:


y1


# In[39]:


#variable to binary values.


y1.loc[y1[13] > 0, [13]] = 1


# In[40]:


y1


# In[41]:


y = y1


# In[42]:


y


# In[43]:


train_labels = np.array(y_train)
train_samples = np.array(y)


# In[44]:


from sklearn.datasets import load_digits

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_labels).reshape(-1,1))


# In[45]:


for i in scaled_train_samples:
    print (i)


# In[46]:


model = Sequential()
model.add(Dense(13, input_shape=(13,), activation='relu'))
model.add(Dense(6, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 


# In[47]:


model.summary()


# In[50]:


#split 
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(nz2, y, test_size=0.2)


# In[51]:


#Dummy variables 

y_train


# In[52]:


#in keras binary_accuracy
from keras.metrics import categorical_crossentropy


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=150, batch_size=10) 
model.summary()
scores = model.evaluate(X_train, y_train)
print("\n%s : %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[53]:


X_train


# In[54]:


y_test


# In[58]:




import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


clf = svm.SVC(kernel = 'rbf', C = 1000, probability = True).fit(X_train, y_train)
y_predicted = clf.predict(X_test)


# In[59]:


y_predicted


# In[60]:


from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report


# In[61]:


#percision and recall for both classes
print(classification_report(y_test, y_predicted))


import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


confusion_matrix(y_test, y_predicted)


# In[62]:


#precision and recall in Keras

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


# In[63]:


precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)


# In[64]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision, recall])


# In[65]:


model.fit(X_train, y_train, nb_epoch=150, batch_size=10) 
model.summary()
scores = model.evaluate(X_train, y_train)
print("\n%s : %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[66]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Plot a confusion matrix.
# cm is the confusion matrix, names are the names of the classes.
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    



# Plot an ROC. pred - the predictions, y - the expected output.
def plot_roc(pred,y):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)


# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)




# In[67]:


#metrics

import numpy as np
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predicted)


# In[68]:


#roc plot

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[73]:


#roc plot more details with auc
from sklearn import metrics
y_pred_proba = model.predict_proba(X_test)[::,]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred_proba)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predicted)

plt.figure()
lw = 2
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("Accuracy", metrics.accuracy_score(y_test, y_predicted))


# In[ ]:


STOP HERE ............................................................................


# In[ ]:




