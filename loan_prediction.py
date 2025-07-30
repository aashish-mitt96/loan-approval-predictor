import pandas as pd
import joblib as dump
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
data = pd.read_csv('data/dataset.csv')
data.columns = data.columns.str.strip()

# Drop ID column
data.drop(columns=['loan_id'], inplace=True)

# Encode categorical features
le_education = LabelEncoder()
le_self_employed = LabelEncoder()

if data['education'].dtype == 'object':
    data['education'] = le_education.fit_transform(data['education'])

if data['self_employed'].dtype == 'object':
    data['self_employed'] = le_self_employed.fit_transform(data['self_employed'])

# Encode target
le_target = LabelEncoder()
data['loan_status'] = le_target.fit_transform(data['loan_status'])

# Create total_assets
data['total_assets'] = (data['residential_assets_value'] + data['commercial_assets_value'] + data['luxury_assets_value'] + data['bank_asset_value']
)
data.drop(columns=['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value','bank_asset_value'], inplace=True)

# Derived features
data['debt_to_income'] = data['loan_amount'] / (data['income_annum'] + 1)
data['loan_to_assets'] = data['loan_amount'] / (data['total_assets'] + 1)

# Features and target
X = data.drop(columns=['loan_status'])
y = data['loan_status']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(x_test))
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save everything
dump.dump(model, 'loan_model.pkl')
dump.dump(scaler, 'scaler.pkl')
