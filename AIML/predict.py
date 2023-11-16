import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
df = df.drop(['Person ID','Occupation','Quality of Sleep','Blood Pressure','Age'] ,axis=1)
label_encoder = preprocessing.LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Gender'].unique()
df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
df['BMI Category'].unique()

X = df.drop(['Sleep Disorder'],axis=1)
y = df['Sleep Disorder']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred_logi = classifier.predict(X_test)
accuracy_score_logi = accuracy_score(y_test, y_pred_logi)
print("accuracy score for logistic regression :", accuracy_score_logi)

#KNN CLassifier
classifier_knn = KNeighborsClassifier() 
classifier_knn.fit(X_train, y_train)
y_pred__knn = classifier_knn.predict(X_test)
accuracy_score_knn = accuracy_score(y_test, y_pred__knn)
print("accuracy score for KNN :", accuracy_score_knn)

#SVM
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train, y_train)
y_pred__svm = model_svm.predict(X_test)
accuracy_score_svm = accuracy_score(y_test, y_pred__svm)
print("accuracy score for svm :", accuracy_score_svm)

#Decision Tress
model_decision = DecisionTreeClassifier(random_state = 0)
model_decision.fit(X_train, y_train)
y_pred__deci = model_decision.predict(X_test)
accuracy_score_decision = accuracy_score(y_test, y_pred__deci)
print("accuracy score for svm :", accuracy_score_decision)

#Random Forest
model_forest = RandomForestClassifier(n_estimators = 100)
model_forest.fit(X_train, y_train)
y_pred__forest = model_forest.predict(X_test)
accuracy_score_forest = accuracy_score(y_test, y_pred__forest)
print("accuracy score for svm :", accuracy_score_forest)

y_pred_final = classifier.predict([[0, 0, 60, 5,0,70,5000 ]])
print(y_pred_final[0])

y_pred_new = model_decision.predict([[0, 3, 30, 8,0,70,2000 ]])
print(y_pred_new[0])


# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3, activation='softmax'))


# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

