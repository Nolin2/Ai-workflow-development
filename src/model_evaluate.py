import joblib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report

# --- Configuration ---
MODEL_PATH = 'models/churn_predictor.pkl'

def evaluate_model():
    """Loads the model and test data to calculate evaluation metrics."""
    
    # --- Load Model and Test Data ---
    try:
        model = joblib.load(MODEL_PATH)
        X_test = joblib.load('data/X_test.pkl')
        y_test = joblib.load('data/y_test.pkl')
    except FileNotFoundError:
        print("Error: Ensure data_prep.py and model_train.py were run successfully.")
        return

    print("Starting model evaluation on the UNSEEN test set...")
    y_pred = model.predict(X_test)

    # --- Evaluation Metrics Calculation ---
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\n--- Model Performance Metrics ---")
    
    # This output matches the required content for your report and Screenshot
    print("\nConfusion Matrix (for Report):")
    print(cm)
    
    print("\nKey Business Metrics:")
    print(f"Precision Score (Efficiency): {precision:.4f}")
    print(f"Recall Score (Coverage - Primary Goal): {recall:.4f}")
    
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == '__main__':
    evaluate_model()
