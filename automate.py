from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,confusion_matrix, ConfusionMatrixDisplay)
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT = "Genetic-Disorders-Classification"
mlflow.set_experiment(EXPERIMENT)

def log_cv_run(name, clf, X, y, skf, smote):
    """One nested run per model: train 5-fold, log mean metrics."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    accs, f1s, aucs = [], [], []
    for tr, va in skf.split(X, y):
        Xtr, ytr = smote.fit_resample(X[tr], y[tr])
        clf.fit(Xtr, ytr)
        yp = clf.predict(X[va])
        ypp = clf.predict_proba(X[va])
        accs.append(accuracy_score(y[va], yp))
        f1s.append(f1_score(y[va], yp, average="weighted"))
        aucs.append(roc_auc_score(y[va], ypp, multi_class="ovr", average="weighted"))

    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(clf.get_params())
        mlflow.log_metrics({
            "mean_accuracy": np.mean(accs),
            "std_accuracy":  np.std(accs),
            "mean_f1":       np.mean(f1s),
            "std_f1":        np.std(f1s),
            "mean_auc":      np.mean(aucs),
            "std_auc":       np.std(aucs),
        })
    return np.mean(f1s)

def run_mlflow_tracking(models, X_pca, y, skf, smote, search, best_model_name,final_model, le_target, pca, scaler, imputer):
    """Call once after the notebook pipeline finishes."""
    with mlflow.start_run(run_name="full-pipeline"):
        mlflow.log_params({
            "pca_components":        pca.n_components_,
            "pca_variance_retained": f"{pca.explained_variance_ratio_.sum():.4f}",
            "smote":                 True,
            "imputer":               "KNNImputer_k5",
            "scaler":                "StandardScaler",
            "n_samples":             X_pca.shape[0],
        })

        for name, clf in models.items():
            log_cv_run(name, clf, X_pca, y, skf, smote)

        mlflow.log_params({f"tuned_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("tuning_best_cv_f1", search.best_score_)

        y_pred = final_model.predict(X_pca)
        y_prob = final_model.predict_proba(X_pca)
        mlflow.log_metrics({
            "final_accuracy": accuracy_score(y, y_pred),
            "final_f1":       f1_score(y, y_pred, average="weighted"),
            "final_auc":      roc_auc_score(y, y_prob, multi_class="ovr", average="weighted"),
        })

        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred, display_labels=le_target.classes_, cmap="Blues", ax=ax
        )
        ax.set_title(f"{best_model_name} (tuned)")
        fig.tight_layout()
        fig.savefig("confusion_matrix.png", dpi=120)
        plt.close(fig)
        mlflow.log_artifact("confusion_matrix.png")

        # -- log final model --
        mlflow.sklearn.log_model(final_model, artifact_path="model")
        print(f"✅ MLflow run logged → experiment: {EXPERIMENT}")

if __name__ != "__main__":
    pass