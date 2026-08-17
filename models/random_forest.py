from preprocessing import *
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

class RandomForestModel:

    def fit(self, X_train_scaled, y_train, parameter_grid):
        rfc_pipeline = Pipeline([
            ("smote", SMOTE(random_state=42)),
            ("rfc", RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ))
        ])

        random_forest_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        grid_search_random_forest = GridSearchCV(
            estimator=rfc_pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=random_forest_cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=1,
            refit=True
        )

        grid_search_random_forest.fit(X_train_scaled, y_train)

        return grid_search_random_forest

    def best_model(self, grid_search_random_forest):
        print("Best Random Forest parameters:")
        print(grid_search_random_forest.best_params_)
        print("Best validation F1-score:",grid_search_random_forest.best_score_)

        best_random_forest = (grid_search_random_forest.best_estimator_)
        return best_random_forest

    def importance(self, grid_search_random_forest, best_random_forest, X_train):
        random_forest_results = pd.DataFrame(grid_search_random_forest.cv_results_)

        top10 = (
            random_forest_results
            .sort_values("mean_test_score", ascending=False)
            .head(10)
        )

        top10 = top10[
            [
                "param_rfc__n_estimators",
                "param_rfc__max_depth",
                "param_rfc__min_samples_split",
                "param_rfc__min_samples_leaf",
                "mean_test_score"
            ]
        ]

        # print(top10)

        importance = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": best_random_forest.named_steps["rfc"].feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        plt.figure(figsize=(10,6))

        plt.barh(
            importance["Feature"],
            importance["Importance"]
        )

        plt.gca().invert_yaxis()

        plt.title("Top 10 Most Important Features")
        plt.xlabel("Feature Importance")

        plt.tight_layout()
        plt.show()

        return top10