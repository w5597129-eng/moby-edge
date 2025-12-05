import lightgbm as lgb
import joblib
import numpy as np
import os

# Create dummy data (100 samples, 10 features)
# 10 features matches the V18 feature extractor output
X = np.random.rand(100, 10)
# 3 classes: 0=Normal, 1=FaultA, 2=FaultB
y = np.random.randint(0, 3, 100)

print("Training dummy LightGBM model...")
clf = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
clf.fit(X, y)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model
model_path = "models/lgbm_classifier.joblib"
joblib.dump(clf, model_path)
print(f"Saved dummy model to {model_path}")

# Verify load
loaded = joblib.load(model_path)
print("Verification load successful.")
print(f"Classes: {loaded.classes_}")
