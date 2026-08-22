import keras
from metrics import KPIs, check_classifier_fit
from preprocessing import *

class MLPModel:
    def transform(self, X_train_pca, X_test_pca, y_train, y_test):
        print("Training features:", X_train_pca.shape)
        print("Testing features:", X_test_pca.shape)
        print("Training labels:", np.asarray(y_train).shape)
        print("Testing labels:", np.asarray(y_test).shape)

        y_train_nn = np.asarray(y_train).astype("float32")
        y_test_nn = np.asarray(y_test).astype("float32")
        X_train_nn = np.asarray(X_train_pca).astype("float32")
        X_test_nn = np.asarray(X_test_pca).astype("float32")

        return X_train_nn, X_test_nn, y_train_nn, y_test_nn

    def fit(self, X_train_nn, y_train_nn, early_stopping, mlp_model):
        mlp_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc")
            ]
        )

        history = mlp_model.fit(
            X_train_nn,
            y_train_nn,
            validation_split=0.20,
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def plot_training_history(self, history):
        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history["loss"],
            label="Training loss"
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation loss"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Binary cross-entropy loss")
        plt.title("MLP Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))

        plt.plot(
            history.history["accuracy"],
            label="Training accuracy"
        )

        plt.plot(
            history.history["val_accuracy"],
            label="Validation accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("MLP Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate(self, mlp_model, history, X_test_nn, y_test_nn, y_test):
        test_results = mlp_model.evaluate(
            X_test_nn,
            y_test_nn,
            verbose=0
        )

        for metric_name, metric_value in zip(
            mlp_model.metrics_names,
            test_results
        ):
            print(f"{metric_name}: {metric_value:.4f}")

        y_probability_mlp = mlp_model.predict(X_test_nn, verbose=0).ravel()
        y_pred_mlp = (y_probability_mlp >= 0.50).astype(int)

        KPIs(y_test, y_pred_mlp, y_probability_mlp, "MLP")
        check_classifier_fit(mlp_model, X_test_nn, y_test_nn, y_pred_mlp, y_probability_mlp, "MLP", neural_network=True)