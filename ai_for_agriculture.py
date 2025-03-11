import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

# Data loading function
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f'Data loaded from {file_path}, shape: {data.shape}')
    return data

# Data preprocessing function
def preprocess_data(data):
    # Handling missing values
    data.fillna(data.mean(), inplace=True)

    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)
    print('Data preprocessing completed.')
    return data

# Feature selection function
def select_features(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    print('Features selected.')
    return X, y

# Train-test split function
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print('Data split into training and testing sets.')
    return X_train, X_test, y_train, y_test

# Model training function
def train_model(X_train, y_train, model_type='linear'):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose 'linear' or 'random_forest'.")
    
    model.fit(X_train, y_train)
    print(f'{model_type.capitalize()} model trained.')
    return model

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Model evaluation completed.\nMSE: {mse}, R2: {r2}')
    return mse, r2

# Save model function
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f'Model saved to {filename}.')

# Load model function
def load_model(filename):
    model = joblib.load(filename)
    print(f'Model loaded from {filename}.')
    return model

# Visualization function
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.show()

# Main function for executing the AI pipeline
def main(data_file, target_column, model_type='linear'):
    data = load_data(data_file)
    data = preprocess_data(data)
    X, y = select_features(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train, model_type)
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    save_model(model, 'model.pkl')
    plot_results(y_test, model.predict(X_test))

# If running as standalone script
if __name__ == "__main__":
    main('agriculture_data.csv', 'yield', 'random_forest')

# Example function to generate synthetic data (for testing purposes)
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'rainfall': np.random.uniform(0, 200, num_samples),
        'temperature': np.random.uniform(20, 35, num_samples),
        'soil_moisture': np.random.uniform(10, 50, num_samples),
        'fertilizer_amount': np.random.uniform(0, 200, num_samples),
        'yield': np.random.uniform(100, 1000, num_samples) 
    }
    df = pd.DataFrame(data)
    df.to_csv('agriculture_data.csv', index=False)
    print('Synthetic data generated and saved to agriculture_data.csv.')

# Uncomment to generate synthetic data
# generate_synthetic_data()

# Include sample usage of the functions provided
if __name__ == "__main__":
    # Uncomment to generate synthetic data for testing
    # generate_synthetic_data()
    
    print("Running the main pipeline...")
    main('agriculture_data.csv', 'yield', 'linear')

# Function to plot feature importances if using Random Forest
def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

# Modified main function to include feature importance plotting
def main_with_importance(data_file, target_column, model_type='linear'):
    data = load_data(data_file)
    data = preprocess_data(data)
    X, y = select_features(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train, model_type)
    mse, r2 = evaluate_model(model, X_test, y_test)
    
    if model_type == 'random_forest':
        plot_feature_importances(model, X.columns)
    
    save_model(model, 'model.pkl')
    plot_results(y_test, model.predict(X_test))

if __name__ == "__main__":
    main_with_importance('agriculture_data.csv', 'yield', 'random_forest')