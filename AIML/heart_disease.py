import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

df = pd.read_csv("heart_2020_cleaned.csv")

df2=df
cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
df2[cols] = df2[cols].apply(LabelEncoder().fit_transform)
df2.drop('SkinCancer',axis=1)
df2.drop('KidneyDisease',axis=1)



x=df2.drop('HeartDisease',axis=1).values
y=df2['HeartDisease'].values

sm = SMOTE(random_state=42)
X, Y = sm.fit_resample(x, y)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30,random_state=42)
rf = RandomForestClassifier(n_estimators= 50 , max_depth= 5 , max_features=12)
rf.fit(x_train , y_train)
rf__ = rf.predict(x_test)

accuracy_score_rf = accuracy_score(y_test, rf__)
print("accuracy score for rf :", accuracy_score_rf)
