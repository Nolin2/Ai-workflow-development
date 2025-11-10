import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
DATA_PATH = 'data/raw/customer_churn_dataset.csv'
TEST_SIZE = 0.20
RANDOM_SEED = 42

def load_and_split_data(file_path):
    """Loads, scales, and splits the data into stratified train and test sets."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}.")
        return None, None, None, None

    print("Data loaded successfully.")
    X = df.drop(['Customer_ID', 'Churn'], axis=1)
    y = df['Churn']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the fitted scaler (CRUCIAL for deployment)
    joblib.dump(scaler, 'models/scaler.pkl')

    # Stratified Split (CRITICAL for imbalanced data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Train/Test split complete. Test set size: {len(X_test)} samples.")
    
    # Save processed data for the next scripts (simulating a pipeline)
    joblib.dump(X_train, 'data/X_train.pkl')
    joblib.dump(X_test, 'data/X_test.pkl')
    joblib.dump(y_train, 'data/y_train.pkl')
    joblib.dump(y_test, 'data/y_test.pkl')
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    load_and_split_data(DATA_PATH)
    print("Processed data and scaler saved.")
