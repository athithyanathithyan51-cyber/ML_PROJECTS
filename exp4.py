from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.images.reshape(-1, 64)
y = OneHotEncoder(sparse_output=False).fit_transform(
    digits.target.reshape(-1, 1)
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(64, activation="relu", input_shape=(64,)),
    Dense(32, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(optimizer="sgd", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

pred = model.predict(X_test, verbose=0).argmax(axis=1)
actual = y_test.argmax(axis=1)

print("Test Accuracy:", round(accuracy_score(actual, pred) * 100, 2), "%")
i = 0
actual = np.argmax(y_test[i])
predicted = np.argmax(model.predict(X_test[i:i+1], verbose=0))

plt.imshow(X_test[i].reshape(8, 8), cmap="gray")
plt.title(f"Actual: {actual} | Predicted: {predicted}")
plt.axis("off")
plt.show() 