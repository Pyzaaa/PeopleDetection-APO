import os
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import joblib
import csv

# ===============================
# CONFIG
# ===============================
CONFIG = {
    "features": {
        "pos_dir": "features_pos",
        "neg_dir": "features_neg"
    },
    "batch_size": 5000,
    "model_out": "hog_8x8-4x4-9-SVC-default.joblib",
    "shuffle": True,
    "n_epochs": 1,
    "log_csv": "learning_curve.csv",
    "sgd": {
        "loss": "hinge",
        "learning_rate": "optimal",
        "random_state": 123,
        "verbose": 1,
        "class_weight": "balanced",
        "max_iter": 50,
    },
    "svc": {
        "random_state": 123,
        "verbose": 1,
        "class_weight": "balanced",
        "max_iter": 50,
    }
}

# ===============================
# Funkcja odczytu batchy z listy plików
# ===============================
def load_feature_batches(file_list, batch_size):
    for i in range(0, len(file_list), batch_size):
        batch_files = file_list[i:i+batch_size]
        if len(batch_files) == 0:
            continue
        batch_features = [np.load(f) for f in batch_files]
        batch_features = np.vstack(batch_features)
        yield batch_features

# ===============================
# Funkcja batchowego treningu z metrykami i automatycznym zapisem
# ===============================
def train_sgd_svm(cfg):
    # clf = SGDClassifier(**cfg["sgd"])
    clf = LinearSVC(**cfg["svc"])
    classes = np.array([0,1])

    pos_files = sorted(glob(os.path.join(cfg["features"]["pos_dir"], "*.npy")))
    neg_files = sorted(glob(os.path.join(cfg["features"]["neg_dir"], "*.npy")))

    if len(pos_files) == 0 or len(neg_files) == 0:
        raise ValueError("Nie znaleziono plików .npy w folderach features_pos lub features_neg!")

    batch_size = min(cfg["batch_size"], len(pos_files), len(neg_files))

    # listy do learning curve
    epoch_acc, epoch_prec, epoch_rec = [], [], []

    # przygotowanie pliku CSV
    with open(cfg["log_csv"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "accuracy", "precision_class1", "recall_class1"])

    print("Rozpoczynam uczenie batchowe...")

    for epoch in range(cfg["n_epochs"]):
        print(f"\n-- Epoch {epoch+1}")

        if cfg["shuffle"]:
            np.random.shuffle(pos_files)
            np.random.shuffle(neg_files)

        pos_gen = load_feature_batches(pos_files, batch_size)
        neg_gen = load_feature_batches(neg_files, batch_size)

        n_batches = min(len(pos_files), len(neg_files)) // batch_size
        if n_batches == 0:
            n_batches = 1

        pbar = tqdm(range(n_batches), desc="Batch", unit="batch")
        for _ in pbar:
            pos_batch = next(pos_gen)
            neg_batch = next(neg_gen)

            y_pos = np.ones(pos_batch.shape[0], dtype=np.int32)
            y_neg = np.zeros(neg_batch.shape[0], dtype=np.int32)

            X_batch = np.vstack([pos_batch, neg_batch])
            y_batch = np.hstack([y_pos, y_neg])

            if cfg["shuffle"]:
                perm = np.random.permutation(len(X_batch))
                X_batch = X_batch[perm]
                y_batch = y_batch[perm]

            clf.fit(X_batch, y_batch)

        # --- zapis modelu ---
        joblib.dump(clf, cfg["model_out"])
        print("Model zapisany:", cfg["model_out"])

        # --- ewaluacja po epoce ---


        print("\n-- ewaluacja po epoce --")
        X_all, y_all = [], []
        for batch in load_feature_batches(pos_files, batch_size):
            X_all.append(batch)
            y_all.append(np.ones(batch.shape[0], dtype=np.int32))
        for batch in load_feature_batches(neg_files, batch_size):
            X_all.append(batch)
            y_all.append(np.zeros(batch.shape[0], dtype=np.int32))
        X_all = np.vstack(X_all)
        y_all = np.hstack(y_all)

        y_pred = clf.predict(X_all)

        acc = accuracy_score(y_all, y_pred)
        prec = precision_score(y_all, y_pred, pos_label=1)
        rec = recall_score(y_all, y_pred, pos_label=1)

        epoch_acc.append(acc)
        epoch_prec.append(prec)
        epoch_rec.append(rec)

        # log w konsoli
        print(f"Epoch {epoch+1}: Accuracy={acc:.4f}, Precision(class1)={prec:.4f}, Recall(class1)={rec:.4f}")

        # zapis do CSV po każdej epoce
        with open(cfg["log_csv"], "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, acc, prec, rec])

        # zapis do .npy
        np.save("learning_curve_acc.npy", np.array(epoch_acc))
        np.save("learning_curve_prec.npy", np.array(epoch_prec))
        np.save("learning_curve_rec.npy", np.array(epoch_rec))

    # --- końcowa ewaluacja ---
    print("\n--- Final Evaluation ---")
    print(classification_report(y_all, y_pred, digits=4))

    # --- rysowanie learning curve ---
    plt.figure(figsize=(8,5))
    plt.plot(range(1, cfg["n_epochs"]+1), epoch_acc, marker='o', label="Accuracy")
    plt.plot(range(1, cfg["n_epochs"]+1), epoch_prec, marker='o', label="Precision (class 1)")
    plt.plot(range(1, cfg["n_epochs"]+1), epoch_rec, marker='o', label="Recall (class 1)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Learning Curve Metrics")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- zapis modelu ---
    joblib.dump(clf, cfg["model_out"])
    print("Model zapisany:", cfg["model_out"])
    return clf

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    clf = train_sgd_svm(CONFIG)
