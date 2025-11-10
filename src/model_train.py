import joblib
from sklearn.ensemble import RandomForestClassifier

# --- Configuration ---
MODEL_PATH = 'models/churn_predictor.pkl'
RANDOM_SEED = 42

def train_and_save_model():
    """Trains the model and saves the artifact."""
    
    # --- Load Data from data_prep.py output ---
    try:
        X_train = joblib.load('data/X_train.pkl')
        y_train = joblib.load('data/y_train.pkl')
    except FileNotFoundError:
        print("Error: Run data_prep.py first to create the processed data files.")
        return

    # --- Model Definition and Training ---
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_SEED, 
        # CRITICAL PARAMETER for imbalanced churn data
        class_weight='balanced',
        max_depth=10 
    )

    print("Starting model training (using balanced class weights)...")
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # --- Save Model ---
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model successfully saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_model()
