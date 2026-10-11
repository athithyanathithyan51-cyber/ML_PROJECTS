import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(url, header=None, names=cols)

print("Dataset shape:", df.shape)
print(df.head())

X = df.drop(columns=["Outcome"])
for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    X.loc[X[c] == 0, c] = X.loc[X[c] != 0, c].median()

X = StandardScaler().fit_transform(X)

k = KMeans(n_clusters=2, random_state=42, n_init=10)
k_labels = k.fit_predict(X)

em = GaussianMixture(n_components=2, random_state=42)
e_labels = em.fit_predict(X)

print("K-Means Silhouette Score:", round(silhouette_score(X, k_labels), 3))
print("EM (GMM) Silhouette Score:", round(silhouette_score(X, e_labels), 3))

X_pca = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels)
plt.title("K-Means Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=e_labels)
plt.title("EM (GMM) Clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")

plt.tight_layout()
plt.show()