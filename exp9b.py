import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("synthetic_multiclass_prostate_cancer.csv")

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# CART with Gini
clf_gini = DecisionTreeClassifier(
    criterion='gini', max_depth=5, min_samples_leaf=5, random_state=42
)
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

# CART with Entropy
clf_entropy = DecisionTreeClassifier(
    criterion='entropy', max_depth=5, min_samples_leaf=5, random_state=42
)
clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Evaluate Gini
evaluate_model("CART using Gini", y_test, y_pred_gini)

# Evaluate Entropy
evaluate_model("CART using Entropy", y_test, y_pred_entropy)

# Visualize Gini Tree
plt.figure(figsize=(20,10))
plot_tree(clf_gini, filled=True, feature_names=X.columns,
          class_names=[str(c) for c in clf_gini.classes_], rounded=True)
plt.title("Decision Tree using Gini")
plt.show()

# Visualize Entropy Tree
plt.figure(figsize=(20,10))
plot_tree(clf_entropy, filled=True, feature_names=X.columns,
          class_names=[str(c) for c in clf_entropy.classes_], rounded=True)
plt.title("Decision Tree using Entropy")
plt.show()
