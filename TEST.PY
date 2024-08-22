import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Download the GTZAN dataset from HuggingFace
# Link: https://huggingface.co/datasets/marsyas/gtzan
# You can download it manually or use the datasets library:
# from datasets import load_dataset
# dataset = load_dataset("marsyas/gtzan")

# For this example, we'll assume you've downloaded and extracted the CSV file
# containing the features. If not, please download it first.

# Load the data
data = pd.read_csv('gtzan_features.csv')

# Prepare the data
X = data.drop(['filename', 'label'], axis=1)
y = data['label']

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a small model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Test accuracy: {accuracy:.2f}")

# You can now use this model to classify music genres
# Remember to preprocess any new data in the same way as the training data
