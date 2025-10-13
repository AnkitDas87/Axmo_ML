import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("synthetic_multiclass_prostate_cancer.csv")

# Features and labels
X = df.drop('target', axis=1).values
y = df['target'].values

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply LDA (keep 2 components for visualization)
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train classifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train_lda, y_train)

# Predict
y_pred = classifier.predict(X_test_lda)

# Evaluate
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", acc)

# Visualize 2D LDA projection
plt.figure(figsize=(8,6))
for label in np.unique(y_train):
    plt.scatter(
        X_train_lda[y_train == label, 0], 
        X_train_lda[y_train == label, 1], 
        label=f"Class {label}", alpha=0.7
    )

plt.title("LDA Projection (Training Data)")
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
