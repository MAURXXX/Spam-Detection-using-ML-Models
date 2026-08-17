from preprocessing import *
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

class LogisticRegressionModel:

    def fit(self, X_train_pca, y_train, parameter_grid):
        logistic_pipeline = Pipeline([
            ("pca", PCA(n_components=2, random_state=42)),
            ("smote", SMOTE(random_state=42)),
            ("logistic", LogisticRegression(
                max_iter=5000,
                random_state=42
            ))
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        grid_search_logistic = GridSearchCV(
            estimator=logistic_pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=1
        )
    
        grid_search_logistic.fit(X_train_pca, y_train)

        return grid_search_logistic

    def best_model(self, grid_search_logistic):
        best_logistic = grid_search_logistic.best_estimator_
        
        print(grid_search_logistic.best_params_)
        print(grid_search_logistic.best_score_)

        return best_logistic    
    
    def eda(self, X_train_pca, y_train):
        plt.figure(figsize=(8,6))

        plt.scatter(
            X_train_pca[:,0],
            X_train_pca[:,1],
            c=y_train,
            cmap="coolwarm",
            edgecolor="k",
            alpha=0.7
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Spam Dataset Before Logistic Regression")
        plt.grid(True)
        plt.show()

    def decision_boundary(self, best_logistic, X_train_scaled, y_train, X_test_scaled):
        fitted_pca = best_logistic.named_steps["pca"]
        fitted_logistic = best_logistic.named_steps["logistic"]

        X_train_2d = fitted_pca.transform(X_train_scaled)
        X_test_2d = fitted_pca.transform(X_test_scaled)

        x_min = X_train_2d[:, 0].min() - 1
        x_max = X_train_2d[:, 0].max() + 1

        y_min = X_train_2d[:, 1].min() - 1
        y_max = X_train_2d[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.05),
            np.arange(y_min, y_max, 0.05)
        )

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        Z = fitted_logistic.predict(grid_points)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(8, 6))

        plt.contourf(
            xx,
            yy,
            Z,
            alpha=0.3,
            cmap="coolwarm"
        )

        plt.scatter(
            X_train_2d[:, 0],
            X_train_2d[:, 1],
            c=y_train,
            cmap="coolwarm",
            edgecolors="black",
            alpha=0.7
        )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Logistic Regression Decision Boundary")
        plt.grid(True)
        plt.show()

        return X_train_2d, X_test_2d

