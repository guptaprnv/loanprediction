import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
dataset=pd.read_csv('loan2.csv')

newdf=dataset.interpolate()
newdf=newdf.fillna(method="ffill")
newdf=newdf.fillna(method="bfill")
X=newdf.iloc[:,1:12]
Y=newdf.iloc[:,[12]]

X=pd.get_dummies(X)
X=X.drop(['Gender_Female','Married_No','Dependents_0','Education_Graduate','Self_Employed_No','Property_Area_Rural'],axis='columns')

Y=pd.get_dummies(Y)
Y=Y.drop(['Loan_Status_N'],axis='columns')

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
Sc_x=StandardScaler()
X_train=Sc_x.fit_transform(X_train)
X_test=Sc_x.transform(X_test)

from sklearn.svm import SVC
classifier=SVC(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

#confusion metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
