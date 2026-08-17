from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from preprocessing import *
from imblearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    GridSearchCV
)
from sklearn.decomposition import PCA
from metrics import KPIs, gmm_silhouette_scorer

class GMMModel:
    def fit(self, X_train_pca, parameter_grid):
        gmm_pipeline = Pipeline([
            ("gmm", GaussianMixture(random_state=42))
        ])

        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        grid_search_gmm = GridSearchCV(
            estimator=gmm_pipeline,
            param_grid=parameter_grid,
            scoring=gmm_silhouette_scorer,
            cv=cv,
            n_jobs=-1,
        )

        grid_search_gmm.fit(X_train_pca)

        return grid_search_gmm

    def best_model(self, grid_search_gmm, X_train_pca, X_test_pca):
        gmm_res = pd.DataFrame(grid_search_gmm.cv_results_)[['param_gmm__n_components', 'param_gmm__covariance_type', 'mean_test_score', 'rank_test_score']]
        best_gmm = grid_search_gmm.best_estimator_
        display(gmm_res.sort_values('rank_test_score').head())
        gmm_results2 = pd.DataFrame(grid_search_gmm.cv_results_)
        
        gmm_results2["Parameters"] = (
            "Components=" +
            gmm_results2["param_gmm__n_components"].astype(str) +
            "\n" +
            gmm_results2["param_gmm__covariance_type"].astype(str)
        )
    
        gmm_results2 = gmm_results2.sort_values(
            by="mean_test_score",
            ascending=False
        )
    
        top_gmm_results = (
            gmm_results2
            .sort_values(
                by="mean_test_score",
                ascending=False
            )
            .head(10)
        )
    
        plt.figure(figsize=(12, 6))
    
        plt.bar(
            top_gmm_results["Parameters"],
            top_gmm_results["mean_test_score"]
        )
    
        plt.xticks(
            rotation=45,
            ha="right"
        )
    
        plt.ylabel("Mean Silhouette Score")
        plt.xlabel("GMM Configuration")
        plt.title("Top 10 GMM Configurations")
        plt.tight_layout()
        plt.show()

        gmm = best_gmm.named_steps["gmm"]
        train_clusters = gmm.predict(X_train_pca)
        test_clusters = gmm.predict(X_test_pca)

        plot_pca = PCA(
            n_components=2,
            random_state=42
        )

        X_train_plot = plot_pca.fit_transform(
            X_train_pca
        )

        return best_gmm, train_clusters, test_clusters, X_train_plot

    def cluster_graphs(self, X_train_plot, train_clusters):

        plt.figure(figsize=(8,6))
        plt.scatter(
            X_train_plot[:,0],
            X_train_plot[:,1],
            s=20,
            alpha=0.6
        )
        plt.title("Spam Dataset Before GMM")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        plt.show()

        plt.figure(figsize=(8,6))

        plt.scatter(
            X_train_plot[:,0],
            X_train_plot[:,1],
            c=train_clusters,
            s=20,
            alpha=0.7,
            cmap="viridis"
        )

        plt.title("Gaussian Mixture Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid()
        plt.show()

    def evaluate(self, best_gmm, X_train_pca, X_test_pca, y_train, y_test, train_clusters, test_clusters):
        unique, counts = np.unique(
            train_clusters,
            return_counts=True
        )

        # cluster_summary = pd.DataFrame({
        #     "Cluster": unique,
        #     "Samples": counts
        # })

        gmm_silhouette_score =  silhouette_score(X_train_pca, train_clusters)
        print("Silhouette Score:", gmm_silhouette_score)
        mapping = {}

        y_train_array = np.asarray(
            y_train
        )

        for cluster in np.unique(train_clusters):
            indices = np.where(train_clusters == cluster)[0]
            majority = np.bincount(y_train_array[indices]).argmax()
            mapping[cluster] = majority

        y_pred_gmm = np.array([mapping[c] for c in test_clusters])

        y_probability_gmm = (
            best_gmm.predict_proba(
                X_test_pca
            )[:, 1]
        )

        gmm_ari = adjusted_rand_score(y_test, test_clusters)
        print("Adjusted Rand Index:", gmm_ari)

        gmm_nmi = normalized_mutual_info_score(y_test, test_clusters)
        print("Normalized Mutual Information:", gmm_nmi)

        KPIs(y_test, y_pred_gmm, y_probability_gmm, "GMM")