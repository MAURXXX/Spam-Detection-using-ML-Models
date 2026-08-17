from preprocessing import *
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

class KNNModel:
    def fit(self, X_train_pca, y_train, parameter_grid):
        pipeline = Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("knn", KNeighborsClassifier())
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        grid_search_knn = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=1
        )

        grid_search_knn.fit(X_train_pca, y_train)

        return grid_search_knn

    def best_model(self, grid_search_knn):
        print("Best parameters:", grid_search_knn.best_params_)
        print(
            "Best cross-validation accuracy:",
            grid_search_knn.best_score_
        )

        results = pd.DataFrame(grid_search_knn.cv_results_)

        # result_table = results[
        #     [
        #         "param_knn__n_neighbors",
        #         "mean_train_score",
        #         "mean_test_score",
        #         "std_test_score"
        #     ]
        # ].sort_values(
        #     by="mean_test_score",
        #     ascending=False
        # )

        return results

    def plot_gridsearch_results(self, results=None):
        
        plot_results = (
            results.groupby("param_knn__n_neighbors", as_index=False)
            .agg(
                mean_train_score=("mean_train_score", "max"),
                mean_test_score=("mean_test_score", "max")
            )
            .sort_values("param_knn__n_neighbors")
        )

        plt.figure(figsize=(9, 5))

        plt.plot(
            plot_results["param_knn__n_neighbors"].astype(int),
            plot_results["mean_train_score"],
            marker="o",
            label="Best training accuracy"
        )

        plt.plot(
            plot_results["param_knn__n_neighbors"].astype(int),
            plot_results["mean_test_score"],
            marker="o",
            label="Best cross-validation accuracy"
        )

        plt.xlabel("Number of neighbours")
        plt.ylabel("Accuracy")
        plt.title("Best KNN Score for Each Neighbour Value")
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate_model(self, X_test_pca, grid_search_knn, results=None):
        best_knn = grid_search_knn.best_estimator_
        best_index = grid_search_knn.best_index_

        best_train_accuracy = results.loc[
            best_index,
            "mean_train_score"
        ]

        best_validation_accuracy = results.loc[
            best_index,
            "mean_test_score"
        ]
            
        y_test_pred = best_knn.predict(X_test_pca)
        y_knn_prob = grid_search_knn.predict_proba(X_test_pca)[:, 1]

        return best_knn, best_train_accuracy, best_validation_accuracy, y_test_pred, y_knn_prob