import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def load_data():
    train = pd.read_csv("sign_mnist_train.csv")
    test = pd.read_csv("sign_mnist_test.csv")
    X = train.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    y = to_categorical(train["label"], num_classes=25)
    Xt = test.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
    yt = to_categorical(test["label"], num_classes=25)
    return X, y, Xt, yt

X, y, Xt, yt = load_data()

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(25, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
history = model.fit(X, y, epochs=10, validation_data=(Xt, yt))
model.save("sign_language_model.h5")

pred = model.predict(Xt, verbose=0).argmax(axis=1)
actual = yt.argmax(axis=1)
print(classification_report(actual, pred, zero_division=0))

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Real-time prediction
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    roi = frame[100:300, 100:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (28, 28)).reshape(1, 28, 28, 1) / 255.0
    letter = chr(np.argmax(model.predict(img, verbose=0)) + 65)
    if letter >= "J":
        letter = chr(ord(letter) + 1)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, "Prediction: " + letter, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()