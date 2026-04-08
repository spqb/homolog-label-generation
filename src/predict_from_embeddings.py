import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse
import os
import time
import h5py
from typing import Optional, Tuple, cast


def get_parser():
    parser = argparse.ArgumentParser(description="Predict from embeddings using multiple classifiers (Logistic Regression, SVM, Random Forest)")
    parser.add_argument("--train_h5", type=str, required=True, help="Path to the training HDF5 file")
    parser.add_argument("--test_h5", type=str, required=True, help="Path to the test HDF5 file")
    parser.add_argument("--output_path", type=str, default="", help="Path to the output file")
    return parser


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    # sklearn classifiers do not accept byte-string class labels.
    if y.dtype.kind == "S":
        return y.astype("U")
    if y.dtype.kind == "O":
        flat = y.reshape(-1)
        if all(isinstance(v, (bytes, bytearray)) for v in flat):
            return np.array([bytes(v).decode("utf-8", errors="replace") for v in flat], dtype="U").reshape(y.shape)
    return y


def _load_embeddings_h5(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as f:
        X_node = f["embeddings"]
        h_node = f["headers"]
        if not isinstance(X_node, h5py.Dataset) or not isinstance(h_node, h5py.Dataset):
            raise ValueError(f"Invalid embeddings file: {path}. 'embeddings' and 'headers' must be datasets.")

        X = cast(np.ndarray, X_node[:])
        h = cast(np.ndarray, h_node[:])

        y: Optional[np.ndarray] = None
        if "labels" in f:
            y_node = f["labels"]
            if not isinstance(y_node, h5py.Dataset):
                raise ValueError(f"Invalid embeddings file: {path}. 'labels' must be a dataset when present.")
            y = cast(np.ndarray, y_node[:])
            y = _normalize_labels(y)
    return X, y, h


def _load_info_h5(path: str) -> str:
    with h5py.File(path, "r") as f:
        if "info" not in f:
            return ""
        info_node = f["info"]
        if not isinstance(info_node, h5py.Dataset):
            raise ValueError(f"Invalid embeddings file: {path}. 'info' must be a dataset when present.")
        info_value = info_node[()]

    if isinstance(info_value, bytes):
        return info_value.decode("utf-8", errors="replace")
    if isinstance(info_value, np.ndarray) and info_value.shape == ():
        scalar = info_value.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8", errors="replace")
        return str(scalar)
    return str(info_value)

def main(args):
    print("Loading data...")
    X_train, y_train, h_train = _load_embeddings_h5(args.train_h5)
    X_test, y_test, h_test = _load_embeddings_h5(args.test_h5)
    info = _load_info_h5(args.train_h5)
    if y_train is None:
        raise ValueError("Training file must contain a 'labels' dataset.")
    
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test headers shape: {h_test.shape}")
    
    # standardize the data
    print("Standardizing data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store predictions
    predictions = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression...")
    start_time = time.time()
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_logreg = logreg.predict(X_test_scaled)
    y_proba_logreg = logreg.predict_proba(X_test_scaled)
    time_logreg = time.time() - start_time
    predictions['logreg'] = {
        'labels_pred': y_pred_logreg,
        'labels_probs': y_proba_logreg
    }
    if y_test is not None:
        acc_logreg = accuracy_score(y_test, y_pred_logreg)
        print(f"Logistic Regression completed - Accuracy: {acc_logreg:.4f} - Time: {time_logreg:.2f}s")
    else:
        print(f"Logistic Regression completed - Time: {time_logreg:.2f}s")
    
    # 2. SVM
    print("\nTraining SVM...")
    start_time = time.time()
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    y_proba_svm = svm.predict_proba(X_test_scaled)
    time_svm = time.time() - start_time
    predictions['SVM'] = {
        'labels_pred': y_pred_svm,
        'labels_probs': y_proba_svm
    }
    if y_test is not None:
        acc_svm = accuracy_score(y_test, y_pred_svm)
        print(f"SVM completed - Accuracy: {acc_svm:.4f} - Time: {time_svm:.2f}s")
    else:
        print(f"SVM completed - Time: {time_svm:.2f}s")
    
    # 3. Random Forest
    print("\nTraining Random Forest...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_proba_rf = rf.predict_proba(X_test_scaled)
    time_rf = time.time() - start_time
    predictions['random_forest'] = {
        'labels_pred': y_pred_rf,
        'labels_probs': y_proba_rf
    }
    if y_test is not None:
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest completed - Accuracy: {acc_rf:.4f} - Time: {time_rf:.2f}s")
    else:
        print(f"Random Forest completed - Time: {time_rf:.2f}s")
    
    # Save results to HDF5
    if args.output_path.split('.')[-1] != 'h5':
        args.output_path += '.h5'
    print(f"\nSaving predictions to {args.output_path}...")
    with h5py.File(args.output_path, 'w') as f:
        f.create_dataset('info', data=np.asarray(info, dtype='S'))
        train_group = f.create_group("train")
        train_group.create_dataset('headers', data=h_train.astype('S'))  # Convert strings to bytes for HDF5
        if y_train is not None:
            train_group.create_dataset('labels_true', data=y_train.astype('S') if y_train.dtype.kind in ('U', 'O') else y_train)

        test_group = f.create_group("test")
        test_group.create_dataset('headers', data=h_test.astype('S'))  # Convert strings to bytes for HDF5
        if y_test is not None:
            test_group.create_dataset('labels_true', data=y_test.astype('S') if y_test.dtype.kind in ('U', 'O') else y_test)

        # Create groups for each classifier
        for classifier_name, preds in predictions.items():
            grp = test_group.create_group(f'predictions/{classifier_name}')
            labels_pred = preds['labels_pred']
            # Convert string labels to bytes if necessary
            if labels_pred.dtype.kind in ('U', 'O'):
                labels_pred = labels_pred.astype('S')
            grp.create_dataset('labels_pred', data=labels_pred)
            grp.create_dataset('labels_probs', data=preds['labels_probs'])
    
    print(f"Successfully saved predictions to {args.output_path}")
    
    if y_test is not None:
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(f"{'Classifier':<20} {'Accuracy':<12} {'Time (s)':<12}")
        print("-"*60)
        print(f"{'Logistic Regression':<20} {acc_logreg:<12.4f} {time_logreg:<12.2f}")
        print(f"{'SVM':<20} {acc_svm:<12.4f} {time_svm:<12.2f}")
        print(f"{'Random Forest':<20} {acc_rf:<12.4f} {time_rf:<12.2f}")
        print("="*60)
    
    print("Done!")
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)