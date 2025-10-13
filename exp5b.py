import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("synthetic_multiclass_prostate_cancer.csv")
X = df.drop("cancer_stage", axis=1)
y = df["cancer_stage"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# Train RBF SVM
model_rbf = SVC(kernel='rbf', C=1.0, gamma=0.01)
model_rbf.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_rbf.predict(X_test)
print("🔸 RBF Kernel SVM Results")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🔍 Plot decision boundaries
def plot_decision_regions(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolors='k', s=40)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    legend_labels = [f'Stage {i}' for i in np.unique(y)]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.grid(True)
    plt.show()

# Plot for RBF SVM
plot_decision_regions(X_pca, y, model_rbf, "SVM with RBF Kernel (PCA Reduced)")
