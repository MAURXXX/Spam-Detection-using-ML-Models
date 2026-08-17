import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class Preprocessing:
    def ingest(self, file_path):
        # Reading Dataset
        data = pd.read_csv(file_path)
        display(data.head())
        return data

    def cleaning(self, data):
        # shape of the dataset
        print("Shape of the dataset:", data.shape)

        # dataset information
        display(data.info())

        # Checking for duplicate values and removing them
        print(f"Duplicate Values: {data.duplicated().sum()}")
        data = data.drop_duplicates().reset_index(drop=True)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # print("Missing values:")
        data.isnull().sum()

        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Checking class distribution
        print(data["spam"].value_counts())
        print(data["spam"].value_counts(normalize=True))

        label_count = data["spam"].value_counts()
        return data, label_count

    def distribution(self, label_count):
        plt.figure(figsize=(6, 4))
        label_count.plot(kind="bar")
        plt.ylabel("Number of records")
        plt.xlabel("Class")
        plt.title("Binary Target Distribution")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def split(self, data):
        # Feature and target separation
        X = data.drop(columns=["spam"])
        y = data["spam"]

        # print(X.dtypes)

        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna()

        y = y.loc[X.index]

        class_names = ["Not Spam", "Spam"]

        # print("Classes:")
        # print("0 = Not Spam")
        # print("1 = Spam")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test, class_names

    def scaling(self, X_train, X_test):
        # Data scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def pca(self, X_train_scaled, X_test_scaled):
        # Applying PCA
        pca_check = PCA()

        pca_check.fit(X_train_scaled)

        cumulative_variance = np.cumsum(
            pca_check.explained_variance_ratio_
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker="o"
        )

        plt.axhline(
            y=0.95,
            linestyle="--",
            label="95% variance"
        )

        plt.xlabel("Number of PCA Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Selecting the Number of PCA Components")
        plt.legend()
        plt.grid()
        plt.show()

        pca = PCA(
            n_components=0.95,
            random_state=42
        )

        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print("Original dimensions:", X_train_scaled.shape)
        print("Reduced dimensions:", X_train_pca.shape)
        print("Total explained variance:", pca.explained_variance_ratio_.sum())

        return X_train_pca, X_test_pca