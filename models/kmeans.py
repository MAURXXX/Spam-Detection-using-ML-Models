from sklearn.model_selection import KFold
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
)
from preprocessing import *
from sklearn.model_selection import (
    KFold,
    GridSearchCV
)
from sklearn.cluster import KMeans


class KMeansModel:
    
    def fit(self, X_train_pca, parameter_grid, scoring_function):
        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        grid_search_kmeans = GridSearchCV(
            estimator=KMeans(
                random_state=42
            ),
            param_grid=parameter_grid,
            scoring=scoring_function,
            cv=cv,
            n_jobs=-1
        )

        grid_search_kmeans.fit(X_train_pca)
        return grid_search_kmeans

    def best_model(self, grid_search_kmeans, X_train_pca):
        best_kmeans = grid_search_kmeans.best_estimator_
        best_k = grid_search_kmeans.best_params_["n_clusters"]

        print("Best Parameters:", grid_search_kmeans.best_params_)
        print("Best CV Silhouette:", grid_search_kmeans.best_score_)
        print("Number of clusters:", best_k)

        print("Best KMeans Inertia:", best_kmeans.inertia_)
        print("Best KMeans Silhouette:", silhouette_score(X_train_pca, best_kmeans.labels_))

        return best_kmeans, best_k

    def cluster_graph(self, best_kmeans, X_train_pca):
        # Reduce PCA data to 2 dimensions for plotting
        plot_pca = PCA(
            n_components=2,
            random_state=42
        )

        # Fit PCA on the training data
        X_train_plot = plot_pca.fit_transform(X_train_pca)

        # Transform K-Means centroids into same 2D space
        centers_plot = plot_pca.transform(best_kmeans.cluster_centers_)
        plt.figure(figsize=(8, 6))

        cluster_labels = best_kmeans.predict(X_train_pca)

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_train_plot[:, 0],
            X_train_plot[:, 1],
            c=cluster_labels,
            alpha=0.5
        )

        plt.scatter(
            centers_plot[:, 0],
            centers_plot[:, 1],
            marker="X",
            s=200,
            edgecolor="black",
            label="Centroids"
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("K-Means Clusters")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # # Plot centroids
        # plt.scatter(
        #     centers_plot[:, 0],
        #     centers_plot[:, 1],
        #     marker="X",
        #     s=250,
        #     c="red",
        #     label="Centroids"
        # )

        # plt.xlabel("Principal Component 1")
        # plt.ylabel("Principal Component 2")
        # plt.title(
        #     f"K-Means Clusters (k = {best_k})"
        # )

        # plt.legend()
        # plt.grid(alpha=0.3)
        # plt.tight_layout()
        # plt.show()

    def evaluate_model(self, best_kmeans, X_test_pca, y_test, X_train_pca, y_train):
        # -----------------------------------
        # Test cluster assignments
        # -----------------------------------
        train_clusters = best_kmeans.predict(X_train_pca)
        cluster_mapping = {}
        y_train_array = np.asarray(y_train)

        # Map each cluster to majority class
        for cluster in np.unique(train_clusters):

            cluster_indices = np.where(train_clusters == cluster)[0]

            majority_class = np.bincount(
                y_train_array[
                    cluster_indices
                ].astype(int)
            ).argmax()

            cluster_mapping[cluster] = majority_class


        print("Cluster Mapping:", cluster_mapping)

        test_clusters = best_kmeans.predict(X_test_pca)

        # Convert cluster IDs into Spam / Not Spam
        y_pred_kmeans = np.array([cluster_mapping[cluster] for cluster in test_clusters])
        # y_probability_kmeans = (best_kmeans.predict_proba(X_test_pca)[:, 1])

        print("Predicted classes:", np.unique(y_pred_kmeans, return_counts=True))
        print("Accuracy:", accuracy_score(y_test, y_pred_kmeans))

        kmeans_ari = adjusted_rand_score(y_test, test_clusters)
        kmeans_nmi = normalized_mutual_info_score(y_test, test_clusters)
        kmeans_silhouette = silhouette_score(X_test_pca, test_clusters)

        print("Adjusted Rand Index:", kmeans_ari)
        print("Normalized Mutual Information:", kmeans_nmi)
        print("Silhouette Score:", kmeans_silhouette)

        # KPIs(y_test, y_pred_kmeans, y_probability_kmeans, "KMeans")