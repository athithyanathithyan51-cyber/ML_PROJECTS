import os, cv2, kagglehub
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

path = kagglehub.dataset_download("emmarex/plantdisease")
root = os.path.join(path, "PlantVillage", "PlantVillage")
X, y = [], []

for folder in os.listdir(root):
    label = "Healthy" if "healthy" in folder.lower() else "Diseased"
    for file in os.listdir(os.path.join(root, folder))[:200]:
        img = cv2.imread(os.path.join(root, folder, file))
        if img is not None:
            X.append(cv2.resize(img, (32, 32)).flatten())
            y.append(label)

X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, pred), 2))
print(classification_report(y_test, pred, labels=["Healthy", "Diseased"]))