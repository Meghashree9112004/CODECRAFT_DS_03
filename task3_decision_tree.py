# task3_decision_tree.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# ---------------------- Style ------------------------
sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.2)
neon_palette = ['#00FFEF', '#FF6EC7', '#FFD700', '#8A2BE2', '#7FFF00']
# ---------------------- Load Dataset ------------------------
df = pd.read_csv("bank.csv", sep=';')

# ---------------------- Basic EDA ------------------------
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nChecking for Missing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['y'].value_counts())

# ---------------------- Encode Categorical Columns ------------------------
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ---------------------- Split Features & Target ------------------------
X = df.drop('y', axis=1)
y = df['y']

# ---------------------- Train-Test Split ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------- Model Training ------------------------
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# ---------------------- Prediction ------------------------
y_pred = clf.predict(X_test)

# ---------------------- Evaluation ------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

# ---------------------- Confusion Matrix Plot ------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm', 
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---------------------- Decision Tree Visualization ------------------------
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=10, impurity=True)
plt.title("Decision Tree - Bank Marketing Dataset")
plt.show()
