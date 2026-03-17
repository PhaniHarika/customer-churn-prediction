import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

engine = create_engine('postgresql://postgres:postgres123@localhost:5432/churn_db')
df = pd.read_sql('SELECT * FROM customers', engine)

# Encode categorical columns
cat_cols = ['gender','partner','dependents','phone_service','multiple_lines',
            'internet_service','online_security','online_backup','device_protection',
            'tech_support','streaming_tv','streaming_movies','contract',
            'paperless_billing','payment_method','churn']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Features and target
X = df.drop(['customer_id','churn'], axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#XG BOOST
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# Save
pickle.dump(model, open('model/churn_model.pkl', 'wb'))
pickle.dump(encoders, open('model/encoders.pkl', 'wb'))
pickle.dump(list(X.columns), open('model/feature_cols.pkl', 'wb'))
print("✅ Model saved!")