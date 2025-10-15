#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[4]:


data = pd.read_csv("Crimes_-_2001_to_Present.csv")
data.head()


# In[5]:


data = data[['Primary Type', 'Date', 'Latitude', 'Longitude']].dropna()
data.head()


# In[6]:


label_encoder = LabelEncoder()
data['Primary Type'] = label_encoder.fit_transform(data['Primary Type'])


# In[13]:


sub_data=data.sample(n=5000,random_state=42)
X = sub_data[['Latitude', 'Longitude']].values
y = sub_data['Primary Type'].values


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# In[22]:


knn = KNN(k=5)
knn.fit(X_train, y_train)


# In[23]:


predictions = knn.predict(X_test)


# In[24]:


accuracy = np.sum(predictions == y_test) / len(y_test)
print("Accuracy:", accuracy)


# In[19]:


predicted_crimes = label_encoder.inverse_transform(predictions[:10])
actual_crimes = label_encoder.inverse_transform(y_test[:10])
print("Predicted crimes:", predicted_crimes)
print("Actual crimes:", actual_crimes)

