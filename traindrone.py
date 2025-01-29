import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

gestures = ["up", "down", "left", "right", "forward", "backward"]
X, y = [], []

for idx, gesture in enumerate(gestures):
    data = np.load(f"gesture_data/{gesture}.npy")
    X.append(data)
    y.extend([idx] * len(data))

X = np.vstack(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))

model.save("gesture_classification_model.h5")
print("Model trained and saved as gesture_classification_model.h5")