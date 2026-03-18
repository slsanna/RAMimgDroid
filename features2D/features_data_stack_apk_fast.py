#!/usr/bin/env python3
# train_and_eval_progress_apkagg.py
import os, csv, math, pickle, argparse, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count, get_context
from functools import partial
from itertools import islice
from tqdm import tqdm

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================
# CONFIG (edit paths)
# =========================
CSV_PATH        = "path/to/image_features_dataset.csv"
CACHE_SHARD_DIR = "path/to/cache_shards"
MODEL_DIR       = "saved_models_R"
RANDOM_STATE    = 42

# Route joblib scratch to a disk with space (avoid /tmp OSError 28)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.path.join(CACHE_SHARD_DIR, "joblib_tmp"))
os.makedirs(os.environ["JOBLIB_TEMP_FOLDER"], exist_ok=True)

FEATURE_COLS = [
    'rgb_hist','hsv_hist','color_moments','glcm_features',
    'lbp_hist','hog_features','edge_stats','hu_moments',
    'stat_features','freq_features'
]

# =========================
# HELPERS
# =========================
def _parse_list_fast(s: str) -> np.ndarray:
    s = s.strip()
    if s and s[0] == '[' and s[-1] == ']':
        s = s[1:-1]
    if not s:
        return np.empty(0, dtype=np.float32)
    return np.fromstring(s, sep=',', dtype=np.float32)

def extract_apk_tag(fp: str) -> str:
    """Robust APK id extraction from file_path."""
    m = re.search(r"/Android/[^/]+/([^/]+)/images/", fp)
    if m: return m.group(1)
    m = re.search(r"/(benign_dumps|malware_dumps)/([^/]+)/images/", fp)
    if m: return m.group(2)
    parts = fp.split("/")
    if "images" in parts:
        i = parts.index("images")
        if i - 1 >= 0 and parts[i-1]:
            return parts[i-1]
    return os.path.basename(os.path.dirname(fp))

def _parse_row(row, feat_idx, i_label, i_path):
    feats = [_parse_list_fast(row[i]) for i in feat_idx]
    x = np.concatenate(feats, dtype=np.float32)
    y = np.int8(row[i_label])
    fp = row[i_path]
    apk_tag = extract_apk_tag(fp)
    return x, y, apk_tag, fp

# =========================
# CSV → SHARDED NPZ (streaming, parallel)
# =========================
def _count_lines(path):
    print("🔎 Counting rows...")
    with open(path, 'r', newline='') as f:
        return max(0, sum(1 for _ in f) - 1)

def _first_vector_dim(path):
    print("🔎 Inferring feature dimension...")
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        feat_idx = [headers.index(c) for c in FEATURE_COLS]
        i_label = headers.index('label')
        i_path  = headers.index('file_path')
        for row in reader:
            x, *_ = _parse_row(row, feat_idx, i_label, i_path)
            if x.size:
                return x.size
    raise RuntimeError("Cannot infer feature dimension from CSV.")

def _read_in_chunks(reader, chunk_size):
    while True:
        chunk = list(islice(reader, chunk_size))
        if not chunk:
            break
        yield chunk

def _process_block(block_rows, feat_idx, i_label, i_path):
    out = [_parse_row(r, feat_idx, i_label, i_path) for r in block_rows]
    X = [o[0] for o in out]
    y = np.fromiter((o[1] for o in out), dtype=np.int8, count=len(out))
    apk = [o[2] for o in out]
    fp  = [o[3] for o in out]
    return X, y, apk, fp

def build_shards(csv_path=CSV_PATH, out_dir=CACHE_SHARD_DIR,
                 chunk_size=50_000, sub_block=2_000, shard_dtype="float32"):
    os.makedirs(out_dir, exist_ok=True)
    n_rows = _count_lines(csv_path)
    dim    = _first_vector_dim(csv_path)
    print(f"rows: {n_rows:,}  dim: {dim}  → shards of ~{chunk_size:,} rows  (dtype={shard_dtype})")

    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        feat_idx = [headers.index(c) for c in FEATURE_COLS]
        i_label = headers.index('label')
        i_path  = headers.index('file_path')

        procs = max(1, cpu_count() - 1)
        worker = partial(_process_block, feat_idx=feat_idx, i_label=i_label, i_path=i_path)
        shard_id = 0
        processed = 0
        total_chunks = math.ceil(n_rows / chunk_size)

        with get_context("spawn").Pool(processes=procs) as pool:
            for chunk in tqdm(_read_in_chunks(reader, chunk_size),
                              total=total_chunks, desc="📥 Building shards"):
                blocks = [chunk[i:i+sub_block] for i in range(0, len(chunk), sub_block)]
                results = pool.imap_unordered(worker, blocks, chunksize=1)

                X_list, ys, apks, fps = [], [], [], []
                for Xc, yc, apkc, fpc in results:
                    X_list.extend(Xc)
                    ys.append(yc)
                    apks.extend(apkc)
                    fps.extend(fpc)

                if not X_list:
                    continue
                dim0 = len(X_list[0])
                if any(len(v) != dim0 for v in X_list):
                    raise ValueError("Inconsistent feature lengths within a shard.")

                X = np.vstack(X_list).astype(shard_dtype, copy=False)
                y = np.concatenate(ys)
                apk_tag = np.array(apks, dtype=object)
                file_path = np.array(fps, dtype=object)

                shard_path = os.path.join(out_dir, f"shard_{shard_id:05d}.npz")
                np.savez_compressed(shard_path, X=X, y=y, apk_tag=apk_tag, file_path=file_path)
                shard_id += 1
                processed += len(X)
                tqdm.write(f"  → wrote {shard_path}  ({len(X):,} rows, dim={X.shape[1]})  total={processed:,}/{n_rows:,}")

    with open(os.path.join(out_dir, "_META.txt"), "w") as f:
        f.write(f"rows={n_rows}\n")
        f.write(f"dim={dim}\n")
        f.write(f"chunk_size={chunk_size}\n")
        f.write(f"dtype={shard_dtype}\n")
    print(" Shards ready at:", out_dir)

# =========================
# LOADING FROM SHARDS (ALL)
# =========================
def load_all_shards(shard_dir=CACHE_SHARD_DIR):
    Xs, ys, apks, fps = [], [], [], []
    shard_files = sorted(p for p in os.listdir(shard_dir) if p.endswith(".npz"))
    for sf in tqdm(shard_files, desc="📦 Loading all shards"):
        d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"]); apks.append(d["apk_tag"]); fps.append(d["file_path"])
    X = np.vstack(Xs)
    y = np.concatenate(ys).astype(int)
    apk = np.concatenate(apks)
    fp  = np.concatenate(fps)
    return X, y, apk, fp

# =========================
# REGION-LEVEL SAMPLER (cap train regions per APK; remaining regions → test)
# =========================
def load_sampled_by_apk(shard_dir=CACHE_SHARD_DIR,
                        per_apk_train=1500, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    per_apk = defaultdict(list)
    shard_files = sorted(p for p in os.listdir(shard_dir) if p.endswith(".npz"))

    for sf in tqdm(shard_files, desc="🧮 Scanning APK indices"):
        d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
        apks = d["apk_tag"]
        for i, a in enumerate(apks):
            per_apk[a].append((sf, i))

    keep = defaultdict(lambda: {"train": [], "test": []})
    for a, lst in per_apk.items():
        lst = list(lst)
        rng.shuffle(lst)
        keep[a]["train"] = lst[:per_apk_train]
        keep[a]["test"]  = lst[per_apk_train:]  # all remaining regions

    def materialize(which):
        by_shard = defaultdict(list)
        for a, slots in keep.items():
            for sf, idx in slots[which]:
                by_shard[sf].append((a, idx))
        Xs, ys, apks, fps = [], [], [], []
        for sf in tqdm(sorted(by_shard), desc=f"📦 Loading {which}"):
            d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
            idxs = [i for _, i in by_shard[sf]]
            Xs.append(d["X"][idxs])
            ys.append(d["y"][idxs])
            apks.extend([a for a, _ in by_shard[sf]])
            fps.extend(d["file_path"][idxs])
        X = np.vstack(Xs) if Xs else np.empty((0, 0), dtype=np.float32)
        y = np.concatenate(ys).astype(int) if ys else np.empty((0,), dtype=int)
        apk = np.array(apks, dtype=object)
        fp  = np.array(fps, dtype=object)
        return X, y, apk, fp

    Xtr, ytr, apk_tr, fp_tr = materialize("train")
    Xte, yte, apk_te, fp_te = materialize("test")
    return Xtr, ytr, apk_tr, fp_tr, Xte, yte, apk_te, fp_te

# =========================
# APK-LEVEL SPLITTER (N APKs per class → train; remaining APKs → test)
# =========================
def load_by_apk_split(
    shard_dir=CACHE_SHARD_DIR,
    train_apks_per_class=1500,
    apk_allowlist_regex=None,
    apk_blocklist_regex=None,
    per_apk_region_cap_train=None,
    per_apk_region_cap_test=None,
    seed=RANDOM_STATE
):
    rng = np.random.default_rng(seed)

    per_apk_indices = defaultdict(list)
    apk_label = {}
    shard_files = sorted(p for p in os.listdir(shard_dir) if p.endswith(".npz"))

    allow_re = re.compile(apk_allowlist_regex) if apk_allowlist_regex else None
    block_re = re.compile(apk_blocklist_regex) if apk_blocklist_regex else None

    for sf in tqdm(shard_files, desc="🧮 Scanning shards for APKs"):
        d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
        apks = d["apk_tag"]; y = d["y"]
        for i, a in enumerate(apks):
            a_str = str(a)
            if allow_re and not allow_re.search(a_str): continue
            if block_re and block_re.search(a_str):     continue
            per_apk_indices[a].append((sf, i))
            apk_label[a] = int(y[i])  # assumes consistent label per APK

    apks_by_class = defaultdict(list)
    for a, lab in apk_label.items():
        apks_by_class[int(lab)].append(a)

    train_apks, test_apks = set(), set()
    for lab, apk_list in apks_by_class.items():
        apk_list = list(apk_list)
        rng.shuffle(apk_list)
        chosen = apk_list[:train_apks_per_class]
        train_apks.update(chosen)
        test_apks.update(set(apk_list) - set(chosen))

    def materialize(apk_set, which="train"):
        by_shard = defaultdict(list)
        for a in apk_set:
            rows = per_apk_indices[a]
            if which == "train" and per_apk_region_cap_train is not None:
                rows = rows[:per_apk_region_cap_train]
            if which == "test" and per_apk_region_cap_test is not None:
                rows = rows[:per_apk_region_cap_test]
            for sf, idx in rows:
                by_shard[sf].append((a, idx))

        Xs, ys, apks, fps = [], [], [], []
        for sf in tqdm(sorted(by_shard), desc=f"📦 Loading {which} APKs"):
            d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
            idxs = [i for _, i in by_shard[sf]]
            Xs.append(d["X"][idxs])
            ys.append(d["y"][idxs])
            apks.extend([a for a, _ in by_shard[sf]])
            fps.extend(d["file_path"][idxs])

        X = np.vstack(Xs) if Xs else np.empty((0, 0), dtype=np.float32)
        y = np.concatenate(ys).astype(int) if ys else np.empty((0,), dtype=int)
        apk = np.array(apks, dtype=object)
        fp  = np.array(fps),  # keep order
        return X, y, apk, np.array([f for f in fps], dtype=object)

    Xtr, ytr, apk_tr, fp_tr = materialize(train_apks, which="train")
    Xte, yte, apk_te, fp_te = materialize(test_apks,  which="test")

    print("APKs per class (train): " + ", ".join(
        f"class {lab}: {sum(a in train_apks for a in apks_by_class[lab])}"
        for lab in sorted(apks_by_class)))
    print("APKs per class (test): " + ", ".join(
        f"class {lab}: {sum(a in test_apks  for a in apks_by_class[lab])}"
        for lab in sorted(apks_by_class)))

    return Xtr, ytr, apk_tr, fp_tr, Xte, yte, apk_te, fp_te

# =========================
# NEW: Directory-aware APK split
# =========================
def load_by_apk_split_from_dirs(
    shard_dir=CACHE_SHARD_DIR,
    malware_dir_substr=None,      # REQUIRED path substring for malware training pool
    benign_dir_substr=None,       # REQUIRED path substring for benign training pool
    n_malware_train=1500,
    n_benign_train=1500,
    train_order="apk",            # 'apk' or 'path' (how to decide "first")
    per_apk_region_cap_train=None,
    per_apk_region_cap_test=None,
    seed=RANDOM_STATE
):
    """
    - Build a mapping from APK -> (label, all shard indices, example file_path).
    - Define training pools by file_path substring match:
        * malware training pool = APKs (label=1) having any region path containing malware_dir_substr
        * benign  training pool = APKs (label=0) having any region path containing benign_dir_substr
    - Select FIRST n_malware_train / n_benign_train APKs from those pools (deterministic order).
    - Test = all remaining APKs in each class (outside the selected train subset).
    """
    assert malware_dir_substr and benign_dir_substr, "Both malware_dir_substr and benign_dir_substr must be provided"

    rng = np.random.default_rng(seed)

    # Gather info
    per_apk_indices = defaultdict(list)
    apk_label = {}
    apk_example_fp = {}
    shard_files = sorted(p for p in os.listdir(shard_dir) if p.endswith(".npz"))

    for sf in tqdm(shard_files, desc="🧮 Scanning shards for APKs"):
        d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
        apks = d["apk_tag"]; y = d["y"]; fps = d["file_path"]
        for i, (a, lab) in enumerate(zip(apks, y)):
            a = str(a); lab = int(lab)
            per_apk_indices[a].append((sf, i))
            apk_label[a] = lab
            # record one example path to use for ordering or filtering
            if a not in apk_example_fp:
                apk_example_fp[a] = str(fps[i])

    # Pools by class restricted to specific directories
    malware_pool = [a for a, lab in apk_label.items()
                    if lab == 1 and malware_dir_substr in apk_example_fp[a]]
    benign_pool  = [a for a, lab in apk_label.items()
                    if lab == 0 and benign_dir_substr in apk_example_fp[a]]

    # Deterministic "first" selection
    if train_order == "apk":
        malware_pool = sorted(malware_pool)
        benign_pool  = sorted(benign_pool)
    else:  # by example path
        malware_pool = sorted(malware_pool, key=lambda a: apk_example_fp[a])
        benign_pool  = sorted(benign_pool,  key=lambda a: apk_example_fp[a])

    train_malware = set(malware_pool[:n_malware_train])
    train_benign  = set(benign_pool[:n_benign_train])

    # Test = all remaining APKs per class (regardless of path)
    test_malware = {a for a, lab in apk_label.items() if lab == 1 and a not in train_malware}
    test_benign  = {a for a, lab in apk_label.items() if lab == 0 and a not in train_benign}

    # Materialize helper
    def materialize(apk_set, which):
        by_shard = defaultdict(list)
        for a in apk_set:
            rows = per_apk_indices[a]
            if which == "train" and per_apk_region_cap_train is not None:
                rows = rows[:per_apk_region_cap_train]
            if which == "test" and per_apk_region_cap_test is not None:
                rows = rows[:per_apk_region_cap_test]
            for sf, idx in rows:
                by_shard[sf].append((a, idx))

        Xs, ys, apks, fps = [], [], [], []
        for sf in tqdm(sorted(by_shard), desc=f"📦 Loading {which} APKs"):
            d = np.load(os.path.join(shard_dir, sf), allow_pickle=True)
            idxs = [i for _, i in by_shard[sf]]
            Xs.append(d["X"][idxs])
            ys.append(d["y"][idxs])
            apks.extend([a for a, _ in by_shard[sf]])
            fps.extend(d["file_path"][idxs])

        X = np.vstack(Xs) if Xs else np.empty((0, 0), dtype=np.float32)
        y = np.concatenate(ys).astype(int) if ys else np.empty((0,), dtype=int)
        apk = np.array(apks, dtype=object)
        fp  = np.array(fps, dtype=object)
        return X, y, apk, fp

    Xtr_m, ytr_m, apk_tr_m, fp_tr_m = materialize(train_malware, which="train")
    Xtr_b, ytr_b, apk_tr_b, fp_tr_b = materialize(train_benign,  which="train")
    Xtr = np.vstack([Xtr_m, Xtr_b]) if Xtr_m.size and Xtr_b.size else (Xtr_m if Xtr_b.size == 0 else Xtr_b)
    ytr = np.concatenate([ytr_m, ytr_b]) if ytr_m.size and ytr_b.size else (ytr_m if ytr_b.size == 0 else ytr_b)
    apk_tr = np.concatenate([apk_tr_m, apk_tr_b]) if apk_tr_m.size and apk_tr_b.size else (apk_tr_m if apk_tr_b.size == 0 else apk_tr_b)
    fp_tr  = np.concatenate([fp_tr_m, fp_tr_b]) if fp_tr_m.size and fp_tr_b.size else (fp_tr_m if fp_tr_b.size == 0 else fp_tr_b)

    Xte_m, yte_m, apk_te_m, fp_te_m = materialize(test_malware, which="test")
    Xte_b, yte_b, apk_te_b, fp_te_b = materialize(test_benign,  which="test")
    Xte = np.vstack([Xte_m, Xte_b]) if Xte_m.size and Xte_b.size else (Xte_m if Xte_b.size == 0 else Xte_b)
    yte = np.concatenate([yte_m, yte_b]) if yte_m.size and yte_b.size else (yte_m if yte_b.size == 0 else yte_b)
    apk_te = np.concatenate([apk_te_m, apk_te_b]) if apk_te_m.size and apk_te_b.size else (apk_te_m if apk_te_b.size == 0 else apk_te_b)
    fp_te  = np.concatenate([fp_te_m, fp_te_b]) if fp_te_m.size and fp_te_b.size else (fp_te_m if fp_te_b.size == 0 else fp_te_b)

    print(f"TRAIN APKs (malware from '{malware_dir_substr}'): {len(train_malware)}  "
          f"| TRAIN APKs (benign from '{benign_dir_substr}'): {len(train_benign)}")
    print(f"TEST APKs (malware remaining): {len(test_malware)}  | TEST APKs (benign remaining): {len(test_benign)}")

    return Xtr, ytr, apk_tr, fp_tr, Xte, yte, apk_te, fp_te

# =========================
# GROUPED SPLIT (optional, if you loaded ALL rows)
# =========================
def grouped_split_by_apk_df(df, test_size=0.2, random_state=RANDOM_STATE):
    X = np.stack(df['features'].values)
    y = df['label'].astype(int).values
    groups = df['apk_tag'].values
    fps = df['file_path'].values
    gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return (X[train_idx], y[train_idx], groups[train_idx], fps[train_idx],
            X[test_idx],  y[test_idx],  groups[test_idx],  fps[test_idx])

# =========================
# APK-LEVEL PROB AGG EVAL (MEAN over regions + FP/FN logging)
# =========================
def evaluate_apk_prob_agg(model, X, y, apk_tags, file_paths,
                          model_name="", verbose=True,
                          list_region_names=False, max_names=25):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        z = model.decision_function(X)
        probs = (z - z.min()) / (z.max() - z.min() + 1e-12)
    else:
        probs = model.predict(X).astype(float)

    apk_scores = defaultdict(list); apk_true = {}; apk_region_details = defaultdict(list)
    for p, lab, apk, fp in zip(probs, y, apk_tags, file_paths):
        apk_scores[apk].append(float(p))
        apk_true[apk] = int(lab)
        apk_region_details[apk].append(os.path.basename(fp))

    all_apks    = list(apk_scores.keys())
    mean_scores = np.array([np.mean(apk_scores[a]) for a in all_apks])
    true_vec    = np.array([apk_true[a] for a in all_apks], dtype=int)

    if len(np.unique(true_vec)) > 1:
        fpr, tpr, thr = roc_curve(true_vec, mean_scores)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_threshold = float(thr[best_idx]) if thr.size else 0.5
        auc   = float(roc_auc_score(true_vec, mean_scores))
        prauc = float(average_precision_score(true_vec, mean_scores))
    else:
        best_threshold = 0.5
        auc = float("nan"); prauc = float("nan")

    pred_vec = (mean_scores > best_threshold).astype(int)

    if verbose:
        print(f"\n--- {model_name} (APK-level, MEAN aggregation) ---")
        print(f"Best threshold (Youden J): {best_threshold:.4f}")
        print(f"APK Accuracy: {(pred_vec == true_vec).mean()*100:.2f}%")
        print(f"APK ROC-AUC: {auc:.4f} | APK PR-AUC: {prauc:.4f}")
        print("Classification Report (APK):")
        print(classification_report(true_vec, pred_vec, target_names=['Benign','Malicious']))
        print("Confusion Matrix (APK):")
        print(confusion_matrix(true_vec, pred_vec))

    fp_count = 0; fn_count = 0
    print("\nMisclassified APKs (mean over regions):")
    for apk, prob, ytrue, yhat in zip(all_apks, mean_scores, true_vec, pred_vec):
        if yhat != ytrue:
            kind = "FP" if yhat == 1 else "FN"
            if kind == "FP": fp_count += 1
            else:            fn_count += 1
            names = apk_region_details[apk]
            print(f"[APK_MISCLASS] APK={apk} | Type={kind} | True={ytrue} | Pred={int(yhat)} "
                  f"| MeanProb={prob:.4f} | Thr={best_threshold:.4f} | Regions={len(names)}")
            if list_region_names:
                shown = ", ".join(sorted(set(names))[:max_names])
                suffix = " ..." if len(names) > max_names else ""
                print(f"  Regions: {shown}{suffix}")
    print(f"\nTotals: FP={fp_count} | FN={fn_count}")

    return {"apk_keys": all_apks, "mean_scores": mean_scores, "true": true_vec,
            "pred": pred_vec, "auc": auc, "prauc": prauc, "best_thr": best_threshold,
            "fp": fp_count, "fn": fn_count}

# =========================
# TRAINING / TUNING / ENSEMBLES
# =========================
def run_hyperparameter_tuning(X_train, y_train, X_test, y_test, apk_test, fp_test, cv_jobs=1, list_regions=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    param_grids = {
        "KNN": {"n_neighbors": [3, 5], "metric": ["euclidean"]},
        "SVM": {"C": [1.0], "gamma": [0.01], "kernel": ["rbf"]},
        "RandomForest": {"n_estimators": [200], "max_depth": [20], "class_weight": [None]},
        "LogisticRegression": {"C": [0.1], "solver": ["saga"], "max_iter": [3000]},
        "MLP": {"hidden_layer_sizes": [(256,), (512,256)], "max_iter": [500], "activation": ["relu"], "early_stopping": [True]}
    }
    model_objs = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        "MLP": MLPClassifier(random_state=RANDOM_STATE)
    }
    scorer = 'roc_auc'
    k_features = min(200, X_train.shape[1])

    for name, clf in tqdm(model_objs.items(), desc="⚙️ Training models"):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(mutual_info_classif, k=k_features)),
            ('classifier', clf)
        ])
        search = RandomizedSearchCV(
            pipe,
            param_distributions={f'classifier__{k}': v for k, v in param_grids[name].items()},
            n_iter=min(5, sum(len(v) for v in param_grids[name].values())),
            scoring=scorer,
            cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=cv_jobs,
            refit=True,
            random_state=RANDOM_STATE
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        out = os.path.join(MODEL_DIR, name.replace(" ", "") + ".pkl")
        with open(out, 'wb') as f:
            pickle.dump(best, f)
        print(f"Saved {name} to {out}")
        evaluate_apk_prob_agg(best, X_test, y_test, apk_test, fp_test, model_name=name, list_region_names=list_regions)

def run_voting_ensemble(X_train, y_train, X_test, y_test, apk_test, fp_test, list_regions=False):
    def load_model(name_no_space):
        path = os.path.join(MODEL_DIR, name_no_space + ".pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)

    knn = load_model('KNN')
    svm = load_model('SVM')
    rf  = load_model('RandomForest')
    lr  = load_model('LogisticRegression')
    mlp = load_model('MLP')

    scaler   = lr.named_steps['scaler']
    selector = lr.named_steps['selector']
    Xtr_sel  = selector.transform(scaler.transform(X_train))
    Xte_sel  = selector.transform(scaler.transform(X_test))

    ensemble = VotingClassifier(
        estimators=[
            ('knn', knn.named_steps['classifier']),
            ('svm', svm.named_steps['classifier']),
            ('rf',  rf.named_steps['classifier']),
            ('lr',  lr.named_steps['classifier']),
            ('mlp', mlp.named_steps['classifier'])
        ],
        voting='soft'
    )
    ensemble.fit(Xtr_sel, y_train)
    evaluate_apk_prob_agg(ensemble, Xte_sel, y_test, apk_test, fp_test, model_name="Voting Ensemble", list_region_names=list_regions)

def run_boosting_models(X_train, y_train, X_test, y_test, apk_test, fp_test, list_regions=False):
    models = {
        'XGBoost':  XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=RANDOM_STATE),
        'LightGBM': LGBMClassifier(random_state=RANDOM_STATE),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    }
    for name, model in tqdm(models.items(), desc="🚀 Boosting"):
        model.fit(X_train, y_train)
        evaluate_apk_prob_agg(model, X_test, y_test, apk_test, fp_test, model_name=name, list_region_names=list_regions)

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Train/eval with APK-level aggregation on sharded cache.")
    parser.add_argument("--build-shards", action="store_true", help="Build NPZ shards from CSV (one-time).")
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per shard when building.")
    parser.add_argument("--sub-block",  type=int, default=2_000,   help="Rows per worker task inside a shard.")
    parser.add_argument("--shard-dtype", type=str, default="float32", choices=["float32","float16"], help="Feature dtype stored in shards.")
    parser.add_argument("--load-all", action="store_true", help="Load ALL shards into RAM (requires lots of memory).")

    parser.add_argument("--split-mode", choices=["apk", "region", "apk_dirs"], default="apk",
                        help="apk: up to N APKs/class; region: cap regions per APK; apk_dirs: pick first N per-class from specific directories.")
    parser.add_argument("--train-apks-per-class", type=int, default=1500, help="(apk mode)")
    parser.add_argument("--apk-allow", type=str, default=None, help="(apk mode) Regex include by APK name.")
    parser.add_argument("--apk-block", type=str, default=None, help="(apk mode) Regex exclude by APK name.")
    parser.add_argument("--cap-train-regions", type=int, default=None, help="Cap regions per APK in TRAIN.")
    parser.add_argument("--cap-test-regions",  type=int, default=None, help="Cap regions per APK in TEST.")

    parser.add_argument("--per-apk-train", type=int, default=1500, help="(region mode) regions per APK for training.")

    # NEW: directory-aware arguments
    parser.add_argument("--train-malware-dir", type=str, default=None, help="Substring path that identifies malware training directory.")
    parser.add_argument("--train-benign-dir",  type=str, default=None, help="Substring path that identifies benign training directory.")
    parser.add_argument("--train-malware-apks", type=int, default=1500, help="#malware APKs from malware dir for training.")
    parser.add_argument("--train-benign-apks",  type=int, default=1500, help="#benign APKs from benign dir for training.")
    parser.add_argument("--train-order", choices=["apk","path"], default="apk", help="How 'first' is decided in apk_dirs mode.")

    parser.add_argument("--cv-jobs", type=int, default=1, help="Parallel jobs for CV; use 1 if disk is tight.")
    parser.add_argument("--list-region-names", action="store_true", help="Print region basenames for misclassified APKs.")
    parser.add_argument("--max-region-names", type=int, default=25, help="Max region names to print when listing.")
    args = parser.parse_args()

    if args.build_shards or not os.path.isdir(CACHE_SHARD_DIR) or not any(p.endswith(".npz") for p in os.listdir(CACHE_SHARD_DIR)):
        print("🔧 Building shards ...")
        build_shards(CSV_PATH, CACHE_SHARD_DIR, chunk_size=args.chunk_size, sub_block=args.sub_block, shard_dtype=args.shard_dtype)

    # Load according to split mode
    if args.load_all:
        print(" Loading ALL shards (ensure you have enough RAM!)")
        X, y, apk, fp = load_all_shards(CACHE_SHARD_DIR)
        df = pd.DataFrame({"features": list(X), "label": y, "apk_tag": apk, "file_path": fp})
        print(f"Loaded {len(df):,} rows")
        X_train, y_train, apk_train, fp_train, X_test, y_test, apk_test, fp_test = grouped_split_by_apk_df(df, test_size=0.2)
    else:
        if args.split_mode == "apk":
            print(" APK-level split: select APKs per class for TRAIN; ALL remaining APKs → TEST")
            X_train, y_train, apk_train, fp_train, X_test, y_test, apk_test, fp_test = load_by_apk_split(
                shard_dir=CACHE_SHARD_DIR,
                train_apks_per_class=args.train_apks_per_class,
                apk_allowlist_regex=args.apk_allow,
                apk_blocklist_regex=args.apk_block,
                per_apk_region_cap_train=args.cap_train_regions,
                per_apk_region_cap_test=args.cap_test_regions,
                seed=RANDOM_STATE
            )
        elif args.split_mode == "apk_dirs":
            print(" APK-level split from specific directories")
            if not args.train_malware_dir or not args.train_benign_dir:
                raise ValueError("In apk_dirs mode, both --train-malware-dir and --train-benign-dir are required.")
            X_train, y_train, apk_train, fp_train, X_test, y_test, apk_test, fp_test = load_by_apk_split_from_dirs(
                shard_dir=CACHE_SHARD_DIR,
                malware_dir_substr=args.train_malware_dir,
                benign_dir_substr=args.train_benign_dir,
                n_malware_train=args.train_malware_apks,
                n_benign_train=args.train_benign_apks,
                train_order=args.train_order,
                per_apk_region_cap_train=args.cap_train_regions,
                per_apk_region_cap_test=args.cap_test_regions,
                seed=RANDOM_STATE
            )
        else:
            print(" Region-level sampling: cap training regions per APK; ALL remaining regions → TEST")
            X_train, y_train, apk_train, fp_train, X_test, y_test, apk_test, fp_test = load_sampled_by_apk(
                CACHE_SHARD_DIR, per_apk_train=args.per_apk_train, seed=RANDOM_STATE
            )

        # Safety fallback if test is empty
        if len(X_test) == 0 or len(np.unique(apk_test)) == 0:
            print(" No test APKs detected after split. Falling back to region-level sampler.")
            X_train, y_train, apk_train, fp_train, X_test, y_test, apk_test, fp_test = load_sampled_by_apk(
                CACHE_SHARD_DIR, per_apk_train=max(200, args.per_apk_train), seed=RANDOM_STATE
            )

        print(f"Train regions: {len(X_train):,} | Test regions: {len(X_test):,} | "
              f"APKs train: {len(np.unique(apk_train)):,} | APKs test: {len(np.unique(apk_test)):,}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # If shards are float16, cast to float32 before fitting (sklearn friendliness)
    if X_train.dtype == np.float16:
        X_train = X_train.astype(np.float32, copy=False)
    if X_test.dtype == np.float16:
        X_test = X_test.astype(np.float32, copy=False)

    # Train & evaluate
    run_hyperparameter_tuning(X_train, y_train, X_test, y_test, apk_test, fp_test,
                              cv_jobs=args.cv_jobs, list_regions=args.list_region_names)
    run_voting_ensemble(X_train, y_train, X_test, y_test, apk_test, fp_test, list_regions=args.list_region_names)
    run_boosting_models(X_train, y_train, X_test, y_test, apk_test, fp_test, list_regions=args.list_region_names)

    print("\n Done.")

if __name__ == "__main__":
    main()
