import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix , classification_report, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('ibm-hr-analytics-attrition-dataset.csv')

df.head()

df.info()

df.describe()

df.isnull().sum()

duplicates = df.duplicated()
duplicates.sum()

df

print("Attrition counts:\n", df['Attrition'].value_counts())

plt.figure(figsize=(8, 6))
plt.pie(df['Attrition'].value_counts(), 
        labels=df['Attrition'].value_counts().index, 
        autopct='%1.1f%%')
plt.title('Attrition Distribution')
plt.show()

df_yes = df[df['Attrition'] == 'Yes']
df_no = df[df['Attrition'] == 'No']

plt.xlabel('Age')
plt.ylabel('MonthlyIncome')
plt.scatter(df_no['Age'], df_no['MonthlyIncome'], color='green', marker='+')
plt.scatter(df_yes['Age'], df_yes['MonthlyIncome'], color='blue', marker='.')
plt.title('Age vs MonthlyIncome (Stayed vs Left)')
plt.show()

le = LabelEncoder()
df['Attrition_n'] = le.fit_transform(df['Attrition'])

X = df[['Age', 'MonthlyIncome', 'JobSatisfaction', 'TotalWorkingYears']]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = SVC()
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test)}")

X_test.shape

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Accuracy: %.2f%%" % (accuracy_train * 100.0))

y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
print("Accuracy: %.2f%%" % (accuracy_train * 100.0))