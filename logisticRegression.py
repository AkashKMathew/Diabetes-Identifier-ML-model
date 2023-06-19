#Data Pre-procesing Step  
# importing libraries  
import numpy as np
import matplotlib.pyplot as mtp  
import pandas as pd  
  
#importing datasets  
import io

data_set = pd.read_csv('diabetes(2).csv')
print(data_set.head(5))


#Changing value=0 to value=mean
data_set['Glucose'] = data_set['Glucose'].replace(0,np.mean(data_set['Glucose']))
data_set['BloodPressure'] = data_set['BloodPressure'].replace(0,np.mean(data_set['BloodPressure']))
data_set['SkinThickness'] = data_set['SkinThickness'].replace(0,np.mean(data_set['SkinThickness']))
data_set['BMI'] = data_set['BMI'].replace(0,np.mean(data_set['BMI']))
data_set['Insulin'] = data_set['Insulin'].replace(0,np.mean(data_set['Insulin']))
print(data_set.head(5))

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, [0,1,2,3,4,5,6,7]].values  
y= data_set.iloc[:, 8].values

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)

#Fitting Logistic Regression to the training set  
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)  

#Predicting the test set result  
y_pred= classifier.predict(x_test)

#Creating the Confusion matrix [[tn,fp],[fn,tp]] 
import tensorflow as tf
from matplotlib import pyplot as plt

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#Finding accuracy = (tp+tn)/total
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("Accuracy=",ac)

#Finding precision = tp/(tp+fp)
from sklearn.metrics import precision_score
pr=precision_score(y_test,y_pred)
print("Precision=",pr)

#Finding recall = tp/(tp+fn)
from sklearn.metrics import recall_score
re=recall_score(y_test,y_pred)
print("Recall=",re)