import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Generate synthetic multiclass dataset
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=42
)

# Create DataFrame and save to CSV
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
csv_filename = "synthetic_multiclass_prostate_cancer.csv"
df.to_csv(csv_filename, index=False)
print(f"Dataset created and saved as {csv_filename}")

# Step 2: Load dataset (from CSV we just saved)
data = pd.read_csv(csv_filename)

# Step 3: Prepare features and labels
X = data.drop('target', axis=1)
y = data['target']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train Decision Tree Classifier using Gini criterion
clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=5, random_state=42)
clf_gini.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = clf_gini.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf_gini, feature_names=feature_names, class_names=[str(i) for i in clf_gini.classes_], filled=True, rounded=True)
plt.title("Decision Tree Classifier (Gini)")
plt.show()
