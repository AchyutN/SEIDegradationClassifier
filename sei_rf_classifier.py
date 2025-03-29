import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Simulate dummy battery dataset
np.random.seed(42)
n_samples = 300

data = pd.DataFrame({
    'sei_thickness': np.random.normal(50, 10, n_samples),
    'sei_growth_rate': np.random.normal(0.5, 0.1, n_samples),
    'avg_ce': np.random.normal(0.98, 0.01, n_samples),
    'delta_impedance': np.random.normal(5, 2, n_samples),
    'capacity_fade_rate': np.random.normal(-0.05, 0.02, n_samples)
})

# 2. Create binary labels (0 = stable SEI, 1 = unstable SEI)
data['label'] = np.where(
    (data['sei_growth_rate'] > 0.55) |
    (data['delta_impedance'] > 6.5) |
    (data['capacity_fade_rate'] < -0.07),
    1, 0
)

# 3. Split data
X = data.drop('label', axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Feature importance
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest: SEI Layer Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
