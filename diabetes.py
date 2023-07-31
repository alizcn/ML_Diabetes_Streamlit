#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# In[30]:


df = pd.read_csv('diabetes.csv') 


# In[31]:


df.head()


# In[32]:


df.describe()


# In[33]:


df.shape


# In[34]:


df['Outcome'].value_counts()


# In[35]:


df.groupby("Outcome").mean()


# In[36]:


y=df.Outcome.values
x_ham_veri=df.drop(["Outcome"],axis=1)
sc=MinMaxScaler()
x_ham_veri=sc.fit_transform(x_ham_veri)
x_train,x_test,y_train,y_test=train_test_split(x_ham_veri,y,test_size=0.1,random_state=1)


# In[39]:


# accuracy score on the training data
X_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print('Accuracy score of the test data : ', test_data_accuracy)


# In[45]:


for i in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    prediction=knn.predict(x_test)
    print(f"K={i} için test verilerimizin dğorulama testi sonucu",knn.score(x_test,y_test))


# In[49]:


new_prediction=knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))


# In[52]:


new_prediction[0]


# In[53]:


import pickle

model_dosyasi="knnmodel.pickle"
pickle.dump(knn,open(model_dosyasi,"wb"))


# In[54]:


scaler_dosyasi="sc.pickle"
pickle.dump(sc,open(scaler_dosyasi,"wb"))


# In[56]:


get_ipython().system('pip install pipreqs')


# In[58]:


get_ipython().system('pipreqs')


# In[ ]:




