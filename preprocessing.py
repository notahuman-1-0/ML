import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Split the dataset into features (X) and target variable (y)
X = data.drop('target_variable', axis=1)
y = data['target_variable']

# Handling missing values
X.fillna(X.mean(), inplace=True)

# Encoding categorical variables
encoder = LabelEncoder()
X['categorical_variable'] = encoder.fit_transform(X['categorical_variable'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
