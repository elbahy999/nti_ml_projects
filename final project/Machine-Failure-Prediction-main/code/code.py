import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('ai4i2020.csv')

cols_to_fix = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all').T

df.drop(['UDI','Product ID'],axis=1,inplace=True)
df.drop(['TWF','HDF','PWF','OSF','RNF'],axis=1,inplace=True)

df.drop(['Type'],axis=1,inplace=True)

numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df.to_csv('cleaned_ai3i2020.csv', index=False)

RandomOverSampler
oversamp = RandomOverSampler(random_state=0)

X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0,stratify=y)

sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

oversamp = RandomOverSampler(random_state=0)
X_train_res, y_train_res = oversamp.fit_resample(X_train_scaled, y_train)

X_train = X_train_res
y_train = y_train_res
X_test = X_test_scaled

model_performance = pd.DataFrame(columns=['Accuracy','Recall','Precision','F1-Score','MCC score','time to train','time to predict','total time'])

model = LogisticRegression(max_iter=1000).fit(X_train_res, y_train_res)

y_predictions = model.predict(X_test_scaled) 

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))

model = DecisionTreeClassifier().fit(X_train,y_train)
y_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))

model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=0,bootstrap=True,).fit(X_train,y_train)
y_predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')

print("Accuracy: "+ "{:.2%}".format(accuracy))
print("Recall: "+ "{:.2%}".format(recall))
print("Precision: "+ "{:.2%}".format(precision))
print("F1-Score: "+ "{:.2%}".format(f1s))

joblib.dump(model, 'random_forest_model.pkl')

joblib.dump(sc, 'scaler.pkl')

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(classification_report(y_train, y_train_pred))

print(classification_report(y_test, y_test_pred))
