#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[3]:


train = pd.read_csv('train_house_kaggle.csv')
test = pd.read_csv('test_house_kaggle.csv')


# In[4]:


# Visualizzare le prime righe del dataset
print(train.head())

# Descrivere le statistiche principali
print(train.describe())

# Controllare i valori mancanti
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])


# In[5]:


# Separare le caratteristiche e il target
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']
test_ids = test['Id']
test = test.drop('Id', axis=1)

# Pipeline per i dati numerici
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline per i dati categoriali
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinare le pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Creare la pipeline del modello
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])


# In[6]:


# Dividere il dataset in train e validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Addestrare il modello
model.fit(X_train, y_train)

# Predire sul validation set
y_pred = model.predict(X_valid)

# Valutare il modello
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'RMSE: {rmse}')


# In[7]:


# Predire sul test set
predictions = model.predict(test)

# Creare il file di sottomissione
output = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
output.to_csv('submission.csv', index=False)
print("Submission file created!")


# In[10]:


# Specifica il percorso dove vuoi salvare il file
file_path = r'D:\Users\pc\Downloads\submission.csv'  # Usa una stringa raw (r'')

# Salva il file CSV
output.to_csv(file_path, index=False)

# Conferma che il file è stato creato
print(f"Submission file created at {file_path}")


# In[ ]:




