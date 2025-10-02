# Titanic Survival Prediction Notebook

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 2. Load Dataset
df = pd.read_csv("C:\\Users\\Benchmatrix WLL\\Downloads\\Nouveau dossier\\Titanic-Dataset.csv")
print(df.head())

# 3. Data Preprocessing
# Drop columns not useful for prediction
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with mode
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])  # male=1, female=0

le_embarked = LabelEncoder()
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# 4. Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Feature Importance (optional)
import matplotlib.pyplot as plt

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance")
plt.show()

from matplotlib.colors import ListedColormap

# Choisir seulement 2 variables pour visualiser
X_vis = df[["Age", "Fare"]]
y_vis = df["Survived"]

# Split pour cohérence
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y_vis, test_size=0.2, random_state=42
)

# Réentraîner un modèle avec seulement Age et Fare
model_vis = RandomForestClassifier(n_estimators=100, random_state=42)
model_vis.fit(X_train_vis, y_train_vis)

# Meshgrid pour représenter l'espace
x_min, x_max = X_vis["Age"].min() - 1, X_vis["Age"].max() + 1
y_min, y_max = X_vis["Fare"].min() - 1, X_vis["Fare"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 5))

Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Moyenne d'âge par groupe (0 = mort, 1 = survivant)
print(df.groupby("Survived")["Age"].mean())

# Visualisation
plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['red','green']))
plt.scatter(X_vis["Age"], X_vis["Fare"], c=y_vis, s=30, cmap=ListedColormap(['red','green']), edgecolor='k')
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Décision du Random Forest (Age vs Fare)")
plt.show()

