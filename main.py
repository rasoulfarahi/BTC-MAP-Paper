def main():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Running training pipeline...")
    model, metrics, y_test, preds = run_training(X, y)

    print("\nResults:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    show_confusion_matrix(y_test, preds)
