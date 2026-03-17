import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:postgres123@localhost:5432/churn_db')

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges before anything
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={
    'customerid'      : 'customer_id',
    'seniorcitizen'   : 'senior_citizen',
    'phoneservice'    : 'phone_service',
    'multiplelines'   : 'multiple_lines',
    'internetservice' : 'internet_service',
    'onlinesecurity'  : 'online_security',
    'onlinebackup'    : 'online_backup',
    'deviceprotection': 'device_protection',
    'techsupport'     : 'tech_support',
    'streamingtv'     : 'streaming_tv',
    'streamingmovies' : 'streaming_movies',
    'paperlessbilling': 'paperless_billing',
    'paymentmethod'   : 'payment_method',
    'monthlycharges'  : 'monthly_charges',
    'totalcharges'    : 'total_charges'
})
print(df.columns.tolist())  # ADD THIS LINE
df = df[['customer_id','gender','senior_citizen','partner','dependents',
         'tenure','phone_service','multiple_lines','internet_service',
         'online_security','online_backup','device_protection','tech_support',
         'streaming_tv','streaming_movies','contract','paperless_billing',
         'payment_method','monthly_charges','total_charges','churn']]

df.to_sql('customers', engine, if_exists='append', index=False)
print(f"✅ Loaded {len(df)} customers to PostgreSQL!")