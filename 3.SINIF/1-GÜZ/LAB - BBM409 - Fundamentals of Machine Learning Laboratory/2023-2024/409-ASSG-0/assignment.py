import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import warnings
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

warnings.filterwarnings('ignore')


df=pd.read_csv("star_classification.csv")
dataset=df.copy()


rows,columns = dataset.shape
#print(f"Number of rows/examples: {rows}")
#print(f"Number of columns/features: {columns}")

#print(f"Examples       {rows}")
#print("---------------------")
#unique=dict(zip(columns,dataset.nunique()))
unique=dataset.nunique()
#print(unique)

unique["Total Rows"] = rows




dataset.drop('rerun_ID',axis=1, inplace=True) 
dataset.drop('spec_obj_ID', axis=1, inplace=True)
print(f"Number of rows/examples: {rows}")
print(f"Number of columns/features: {columns}")


y = dataset['class']
x = dataset.drop('class', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
print('Original dataset shape %s' % Counter(y))
print('Original ytrain dataset shape %s' % Counter(y_train))
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
print('Resampled ytrain dataset shape %s' % Counter(y_train_smote))

rf = RandomForestClassifier(max_depth=7 , max_features=3,n_estimators= 100)
rf.fit(x_train_smote, y_train_smote )

plt.show()
