# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Load the dataset
def load_dataset(file_path):
    """
    Load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

# Step 2: Preprocess the data
def preprocess_data(data):
    """
    Preprocess the dataset by selecting features and target variable.
    """
    # Features: fan_in, fan_out, optimization_scope
    features = ['fan_in', 'fan_out', 'optimization_scope']
    X = data[features]
    
    # Target: logic_depth
    y = data['logic_depth']
    
    return X, y

# Step 3: Train the model
def train_model(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Absolute Error (MAE).
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    return y_pred

# Main function
def main():
    # Step 1: Load the dataset
    file_path = 'rtl_dataset.csv'
    data = load_dataset(file_path)
    
    # Step 2: Preprocess the data
    X, y = preprocess_data(data)
    
    # Step 3: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 4: Train the model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save the model (optional)
    import joblib
    joblib.dump(model, 'logic_depth_predictor.pkl')
    print("Model saved as 'logic_depth_predictor.pkl'")

# Run the program
if __name__ == "__main__":
    main()