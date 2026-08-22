from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def check_classifier_fit(modelf, Xf_train, yf_train, Xf_evaluation, yf_evaluation, model_namef="Model", gap_thresholdf=0.05,
                        low_score_thresholdf=0.70, neural_network=False):

    train_predictions = modelf.predict(Xf_train)
    evaluation_predictions = modelf.predict(Xf_evaluation)

    # Keras models return probabilities,
    # so convert them to binary class predictions
    if neural_network:
        train_predictions = (train_predictions.flatten() >= 0.5).astype(int)
        evaluation_predictions = (evaluation_predictions.flatten() >= 0.5).astype(int)

    train_accuracy = accuracy_score(yf_train, train_predictions)
    evaluation_accuracy = accuracy_score(yf_evaluation, evaluation_predictions)

    train_precision = precision_score(yf_train, train_predictions, zero_division=0)
    evaluation_precision = precision_score(yf_evaluation, evaluation_predictions, zero_division=0)

    train_recall = recall_score(yf_train, train_predictions, zero_division=0)
    evaluation_recall = recall_score(yf_evaluation, evaluation_predictions, zero_division=0)

    train_f1 = f1_score(yf_train,train_predictions,zero_division=0)
    evaluation_f1 = f1_score(yf_evaluation,evaluation_predictions,zero_division=0)

    f1_gap = train_f1 - evaluation_f1

    if (train_f1 < low_score_thresholdf and evaluation_f1 < low_score_thresholdf):
        diagnosis = "Possible underfitting"

    elif f1_gap > gap_thresholdf:
        diagnosis = "Possible overfitting"

    elif evaluation_f1 > train_f1 + gap_thresholdf:
        diagnosis = ("Evaluation score unexpectedly exceeds training score; check preprocessing or data leakage")

    else:
        diagnosis = "Good generalisation"

    results = pd.DataFrame({
        "Model": [model_namef],
        "Train Accuracy": [train_accuracy],
        "Evaluation Accuracy": [evaluation_accuracy],
        "Train Precision": [train_precision],
        "Evaluation Precision": [evaluation_precision],
        "Train Recall": [train_recall],
        "Evaluation Recall": [evaluation_recall],
        "Train F1": [train_f1],
        "Evaluation F1": [evaluation_f1],
        "F1 Difference": [f1_gap],
        "Diagnosis": [diagnosis]
    })

    return results.round(4)

def KPIs(ys_test, ys_pred, y_prob, model):
    class_names = ["Not Spam", "Spam"]
    print(
        model + " test accuracy:",
        accuracy_score(
            ys_test,
            ys_pred
        )
    )

    print(
        classification_report(
            ys_test,
            ys_pred,
            labels=[0, 1],
            target_names=class_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        ys_test,
        ys_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot()
    plt.title(model + " Confusion Matrix")
    plt.show()

    auc = roc_auc_score(
        ys_test,
        y_prob
    )

    print(model + " ROC-AUC:", auc)

    RocCurveDisplay.from_predictions(
        ys_test,
        y_prob
    )

    plt.title(model + " ROC Curve")
    plt.show()

def plot_gridsearch_results(grid_search, parameter, title, x_label, score_label="Mean CV Score", top_n=None):
    """
    Plot GridSearchCV results for one hyperparameter.

    Parameters
    ----------
    grid_search : fitted GridSearchCV
    parameter : str
        Parameter name, e.g. "knn__n_neighbors"
    title : str
        Graph title
    x_label : str
        X-axis label
    score_label : str
        Y-axis label
    top_n : int or None
        Show only top N configurations if specified
    """

    results = pd.DataFrame(
        grid_search.cv_results_
    )

    parameter_column = (
        "param_" + parameter
    )

    plot_data = results[
        [
            parameter_column,
            "mean_test_score"
        ]
    ].copy()

    # Average results if the same parameter
    # appears in multiple configurations
    plot_data = (
        plot_data
        .groupby(parameter_column)
        ["mean_test_score"]
        .mean()
        .reset_index()
    )

    plot_data = plot_data.sort_values(
        "mean_test_score",
        ascending=False
    )

    if top_n is not None:
        plot_data = plot_data.head(top_n)

    ax = plot_data.plot(
        x=parameter_column,
        y="mean_test_score",
        kind="bar",
        figsize=(8, 5),
        legend=False
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.3f",
            padding=3
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(score_label)

    plt.xticks(rotation=0)
    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()

    return plot_data

def gmm_silhouette_scorer(estimator, X, y=None):

    labels = estimator.predict(X)

    if len(np.unique(labels)) < 2:
        return -1

    return silhouette_score(X, labels)

def dbscan_scorer(estimator, X, y=None):

        labels = estimator.fit_predict(X)

        non_noise = labels != -1
        clusters = np.unique(labels[non_noise])

        # Invalid clustering
        if len(clusters) < 2:
            return -1

        if non_noise.sum() <= len(clusters):
            return -1

        silhouette = silhouette_score(
            X[non_noise],
            labels[non_noise]
        )

        coverage = np.mean(non_noise)

        # Penalise solutions that classify
        # most samples as noise
        score = silhouette * coverage

        return score


