import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="wine_class")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convert to DataFrame
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["wine_class"] = y

# 🟢 Plot 1: PC1 vs Wine Class
plt.figure(figsize=(8, 4))
sns.stripplot(x="wine_class", y="PC1", data=pca_df, jitter=True, palette="Set1", alpha=0.7)
plt.title("🔍 PC1 Distribution per Wine Class")
plt.xlabel("Wine Class")
plt.ylabel("PC1")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔵 Plot 2: PC2 vs Wine Class
plt.figure(figsize=(8, 4))
sns.stripplot(x="wine_class", y="PC2", data=pca_df, jitter=True, palette="Set2", alpha=0.7)
plt.title("🔍 PC2 Distribution per Wine Class")
plt.xlabel("Wine Class")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔴 Plot 3: PC1 vs PC2 (Final PCA Scatterplot)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="wine_class", palette="Set1", s=60)
plt.title("🎯 PCA Scatter Plot (PC1 vs PC2)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.grid(True)
plt.legend(title="Wine Class")
plt.tight_layout()
plt.show()

# 🧠 Train/Test Split on PCA Data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

# 🔧 Train SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📊 Evaluation
print("\n📈 PCA Explained Variance:")
print(f"PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

print("\n🎯 Classification Report (SVM on PCA):")
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
