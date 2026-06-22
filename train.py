from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classifier import build_model
from pca_reduction import apply_pca

def run_training(X, y):
    # 1. Apply PCA for dimensionality reduction
    X_reduced = apply_pca(X)
    
    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )
    
    # 3. Initialize the model
    model = build_model()
    
    # 4. Train the model
    model.fit(X_train, y_train)
    
    # 5. Predict and evaluate
    preds = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1-score": f1_score(y_test, preds)
    }
    
    return model, metrics, y_test, preds
