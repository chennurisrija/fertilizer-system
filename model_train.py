import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# load dataset
data = pd.read_csv("crop_dataset.csv")

# features (matching what the API provides: N, P, K, ph)
X = data[['N','P','K','ph']]

# output
y = data['label']

# train model
model = RandomForestClassifier(n_estimators=100)

model.fit(X,y)

# save model
joblib.dump(model,'crop_model.pkl')

print("Model trained successfully")