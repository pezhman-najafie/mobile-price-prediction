# Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load Data
df = pd.read_csv("mobile_dataset.csv")

# Data Exploration
# Uncomment the following lines if you want to visualize the data distribution
# import seaborn as sns
# sns.displot(df)



# print(df.isnull().sum())
# df['column_name'].fillna(value, inplace=True)

# Apply Label Encoding
label_encoder = LabelEncoder()
df['blue'] = label_encoder.fit_transform(df['blue'])
df['sim type'] = label_encoder.fit_transform(df['sim type'])
df['four_g'] = label_encoder.fit_transform(df['four_g'])
df['three_g'] = label_encoder.fit_transform(df['three_g'])
df['type'] = label_encoder.fit_transform(df['type'])
df['wifi'] = label_encoder.fit_transform(df['wifi'])
df['price_range'] = label_encoder.fit_transform(df['price_range'])

column_means = df.mean()
df.fillna(column_means, inplace=True)

# Data Cleaning
# Remove missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Extract Features and Labels
rows, cols = df.shape
# print("rows : {} cols :{} ".format(rows,cols))
data = df.iloc[:, 1:21]
label = df.iloc[:, 21]

# print(df['price_range'])
# print(df['type'] )
# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)


# Standardize Data
scaler1 = StandardScaler()

# Fit and transform on training data
X_train = scaler1.fit_transform(X_train)
X_test = scaler1.transform(X_test)


# KNN Model
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy: {:.2f}".format(acc * 100))

# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy: {:.2f}%".format(dt_acc * 100))

# Random Forest
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy: {:.2f}%".format(rf_acc * 100))

# Support Vector Machine (SVM)
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
print("SVM Accuracy: {:.2f}%".format(svm_acc * 100))

# Logistic Regression
lr_clf = LogisticRegression(solver='liblinear', max_iter=1000)
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy: {:.2f}%".format(lr_acc * 100))

# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix

# KNN Model
knn_pred = clf.predict(X_test)
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))

# Decision Tree
dt_pred = dt_clf.predict(X_test)
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))

# Random Forest
rf_pred = rf_clf.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# Support Vector Machine (SVM)
svm_pred = svm_clf.predict(X_test)
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

# Logistic Regression
lr_pred = lr_clf.predict(X_test)
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
