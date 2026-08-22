from sklearn.cluster import DBSCAN
from sklearn.model_selection import KFold
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    classification_report
)
from preprocessing import *
from sklearn.model_selection import (
    KFold,
    GridSearchCV
)

class DBSCANModel:

    def fit(self, X_train_pca, parameter_grid, scoring_function):
        

        dbscan_cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        grid_search_dbscan = GridSearchCV(
            estimator=DBSCAN(),
            param_grid=parameter_grid,
            scoring=scoring_function,
            cv=dbscan_cv,
            n_jobs=-1
        )

        grid_search_dbscan.fit(X_train_pca)

        return grid_search_dbscan

    def best_model(self, grid_search_dbscan, X_train_pca):
        best_dbscan = grid_search_dbscan.best_estimator_
        best_clusters = best_dbscan.fit_predict(X_train_pca)

        print("Best Parameters:", grid_search_dbscan.best_params_)
        print("Best CV Silhouette:", grid_search_dbscan.best_score_)
        print("Number of clusters:", len(set(best_clusters) - {-1}))

        labels = best_dbscan.labels_
        n_clustered = np.sum(labels >= 0)
        coverage_percentage = (n_clustered / len(labels)) * 100

        print(
            "Coverage:",
            coverage_percentage
        )

        n_noise = list(labels).count(-1)
        noise_percentage = (n_noise / len(labels)) * 100

        print(
            "Noise:",
            noise_percentage
        )

        return best_dbscan, best_clusters

    def evaluate_model(self, best_dbscan, best_clusters, X_train_pca):
        # Number of clustered and noise samples
        clustered_samples = np.sum(best_clusters != -1)
        noise_samples = np.sum(best_clusters == -1)
        total_samples = len(best_clusters)

        # Convert to percentages
        coverage_percentage = (clustered_samples / total_samples) * 100
        noise_percentage = (noise_samples / total_samples) * 100

        print("Cluster Coverage:", round(coverage_percentage, 2), "%")
        print("Noise Percentage:", round(noise_percentage, 2),"%")

        # Graph
        coverage_results = pd.DataFrame({
            "Category": [
                "Cluster Coverage",
                "Noise"
            ],
            "Percentage": [
                coverage_percentage,
                noise_percentage
            ]
        })

        ax = coverage_results.plot(
            x="Category",
            y="Percentage",
            kind="bar",
            figsize=(7, 5),
            legend=False
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.2f%%",
                padding=3
            )

        plt.title("DBSCAN Cluster Coverage vs Noise")
        plt.xlabel("")
        plt.ylabel("Percentage of Samples")
        plt.ylim(0, 100)
        plt.xticks(rotation=0)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        # labels = best_dbscan.labels_
        # n_clustered = np.sum(labels >= 0)
        # coverage_percentage = (n_clustered / len(labels)) * 100

        # print(
        #     "Coverage:",
        #     coverage_percentage
        # )

        # n_noise = list(labels).count(-1)
        # noise_percentage = (n_noise / len(labels)) * 100

        # print(
        #     "Noise:",
        #     noise_percentage
        # )

        visual_pca = PCA(
            n_components=2,
            random_state=42
        )

        X_train_plot = visual_pca.fit_transform(
            X_train_pca
        )

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_train_plot[:, 0],
            X_train_plot[:, 1],
            s=20,
            alpha=0.6
        )

        plt.xlabel("Visual principal component 1")
        plt.ylabel("Visual principal component 2")
        plt.title("Spam Data Before DBSCAN")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 6))

        plt.scatter(
            X_train_plot[:, 0],
            X_train_plot[:, 1],
            c=best_clusters,
            s=20,
            alpha=0.7
        )

        plt.xlabel("Visual principal component 1")
        plt.ylabel("Visual principal component 2")
        plt.title("Spam Data After DBSCAN")
        plt.grid(True)
        plt.show()

    def metrics(self, best_clusters, y_train, X_train_pca):
        cluster_mapping = {}

        y_train_array = np.asarray(y_train)

        for cluster in np.unique(best_clusters):

            if cluster == -1:
                continue

            cluster_indices = np.where(best_clusters == cluster)[0]
            majority_class = np.bincount(y_train_array[cluster_indices].astype(int)).argmax()
            cluster_mapping[cluster] = majority_class

        print("Cluster mapping:", cluster_mapping)

        non_noise_mask = best_clusters != -1
        mapped_predictions = np.array([cluster_mapping[cluster] for cluster in best_clusters[non_noise_mask]])
        actual_non_noise = y_train_array[non_noise_mask]

        print("Accuracy on non-noise records:", accuracy_score(actual_non_noise, mapped_predictions))

        # Remove noise points
        X_non_noise = X_train_pca[non_noise_mask]
        clusters_non_noise = best_clusters[non_noise_mask]

        # Silhouette score requires at least 2 clusters
        if len(np.unique(clusters_non_noise)) > 1:

            dbscan_silhouette = silhouette_score(
                X_non_noise,
                clusters_non_noise
            )

            print("DBSCAN Silhouette Score:", dbscan_silhouette)

        else:
            dbscan_silhouette = np.nan
            print("Silhouette Score cannot be calculated (only one cluster found).")

        dbscan_ari = adjusted_rand_score(y_train, best_clusters)
        print("Adjusted Rand Index:", dbscan_ari)

        dbscan_nmi = normalized_mutual_info_score(y_train, best_clusters)
        print("Normalized Mutual Information:", dbscan_nmi)

        print("Percentage evaluated:", non_noise_mask.mean() * 100)

        return actual_non_noise, mapped_predictions

        # KPIs(
        #     actual_non_noise,
        #     mapped_predictions,
        #     best_clusters[non_noise_mask],
        #     "DBSCAN"
        # )
        # print(
        #     classification_report(
        #         actual_non_noise,
        #         mapped_predictions,
        #         target_names=[
        #             "Not Spam",
        #             "Spam"
        #         ],
        #         zero_division=0
        #     )
        # )