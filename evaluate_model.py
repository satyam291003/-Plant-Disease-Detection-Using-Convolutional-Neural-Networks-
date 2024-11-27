import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Load the preprocessed data (same as during training)
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Split data into validation and test sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, resize the images as done during training (to match input shape)
X_val = np.array([resize(img, (64, 64)) for img in X_val])

# Load the trained model
model = load_model("final_model.keras")

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(X_val, y_val, batch_size=32)

# Print the evaluation results
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Predictions on the validation set
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Optional: Plot confusion matrix as a heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

print("Evaluation complete!")
