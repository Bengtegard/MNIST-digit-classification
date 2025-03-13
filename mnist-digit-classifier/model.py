import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from scipy.ndimage import shift

def load_data():
    """Load the MNIST dataset and split into train and test sets."""
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X, y = mnist.data, mnist.target.astype(np.uint8)
    return X[:60000], y[:60000], X[60000:], y[60000:]

def shift_image(image, dx, dy):
    """Shift the image by dx and dy pixels (data augmentation)."""
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

def augment_data(X_train, y_train):
    """Perform data augmentation by shifting images in 4 directions."""
    X_train_augmented, y_train_augmented = list(X_train), list(y_train)
    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        for image, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image, dx, dy))
            y_train_augmented.append(label)
    return np.array(X_train_augmented), np.array(y_train_augmented)

def train_xgb(X_train, y_train):
    """Train an XGBoost classifier with optimized hyperparameters."""
    best_params = {
        'subsample': 1.0,
        'reg_lambda': 0.01,
        'reg_alpha': 0.1,
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'colsample_bytree': 0.6,
        'objective': 'multi:softprob',
        'num_class': 10,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }

    xgb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(**best_params))
    ])

    print("Training XGBoost model...")
    xgb_pipeline.fit(X_train, y_train)
    return xgb_pipeline

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    # Train model with best hyperparameters
    final_model = train_xgb(X_train_aug, y_train_aug)

    # Save the trained model
    joblib.dump(final_model, "final_xgb_model.pkl")
    print("Model saved as 'final_xgb_model.pkl'.")

    # Evaluate model on test set
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {acc:.4f}")
