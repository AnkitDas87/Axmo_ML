import pandas as pd
import numpy as np  # ✅ Required before using np.random
num_samples = 500
num_features = 500
np.random.seed(42)

# ✅ FIXED: Now generates exactly 500 feature names
feature_names = ['psa_level', 'tumor_size', 'age', 'genetic_score'] + [f'feature_{i}' for i in range(1, num_features - 3)]

X = np.random.normal(0, 1, size=(num_samples, num_features))

# Add realistic patterns
X[:, 0] = np.random.normal(4, 2, num_samples)
X[:, 1] = np.random.normal(2.5, 1, num_samples)
X[:, 2] = np.random.randint(40, 80, num_samples)
X[:, 3] = np.random.normal(50, 15, num_samples)

# Generate labels
risk = 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.1 * X[:, 2] + 0.2 * X[:, 3]
percentiles = np.percentile(risk, [20, 40, 60, 80])
y = np.digitize(risk, percentiles)

# ✅ No mismatch now
df = pd.DataFrame(X, columns=feature_names)
df['cancer_stage'] = y
df.to_csv("synthetic_multiclass_prostate_cancer.csv", index=False)
print("✅ Fixed: Dataset saved.")
