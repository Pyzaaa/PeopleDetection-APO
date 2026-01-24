import os
import cv2
import pandas as pd
import numpy as np
import joblib
import argparse

from skimage.feature import hog
from utils import nms
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, mean_absolute_error


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    return inter / ua


def to_boxes(df):
    cols = {c.lower(): c for c in df.columns}
    if all(k in cols for k in ["xmin", "ymin", "xmax", "ymax", "filename"]):
        x1, y1, x2, y2 = [cols[k] for k in ["xmin", "ymin", "xmax", "ymax"]]
        fcol = cols["filename"]
        boxes = df[[fcol, x1, y1, x2, y2]].copy()
        boxes.columns = ["filename", "xmin", "ymin", "xmax", "ymax"]
        return boxes

    if all(k in cols for k in ["x", "y", "width", "height", "filename"]):
        fcol = cols["filename"]
        x, y, w, h = [cols[k] for k in ["x", "y", "width", "height"]]
        tmp = df[[fcol, x, y, w, h]].copy()
        tmp["xmin"] = tmp[x]
        tmp["ymin"] = tmp[y]
        tmp["xmax"] = tmp[x] + tmp[w]
        tmp["ymax"] = tmp[y] + tmp[h]
        return tmp[[fcol, "xmin", "ymin", "xmax", "ymax"]]

    raise ValueError("Nieznany format CSV")


def pyramid(gray, scale=1.25, min_size=(128, 128)):
    yield gray, 1.0
    inv = 1.0
    cur = gray
    while True:
        h, w = cur.shape[:2]
        w2, h2 = int(w / scale), int(h / scale)
        if h2 < min_size[1] or w2 < min_size[0]:
            break
        cur = cv2.resize(cur, (w2, h2))
        inv *= (1.0 / scale)
        yield cur, inv


def sliding(gray, step=12, win=(64, 128)):
    H, W = gray.shape[:2]
    for y in range(0, H - win[1] + 1, step):
        for x in range(0, W - win[0] + 1, step):
            yield x, y, gray[y:y + win[1], x:x + win[0]]


def detect_img(gray, clf, score_thr=1.9, scale=1.25, step=8, win=(64, 128)):
    boxes, scores = [], []

    for scaled, inv in pyramid(gray, scale=scale, min_size=(128, 128)):
        for x, y, patch in sliding(scaled, step=step, win=win):
            f = hog(
                patch,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                orientations=9,
                block_norm="L2-Hys",
                feature_vector=True,
            )
            s = clf.decision_function([f])[0]
            if s < score_thr:
                continue

            sx, sy = int(x / inv), int(y / inv)
            ex, ey = int((x + win[0]) / inv), int((y + win[1]) / inv)
            boxes.append([sx, sy, ex, ey])
            scores.append(float(s))

    return boxes, scores


def main(
    split,
    iou_thr=0.5,
    nms_thr=0.2,
    score_thr=1.9,
    model_path="hog_svm.joblib",
    limit=10,
):
    im_dir = f"data/{split}/images"
    csv_path = f"data/{split}/annotations.csv"

    clf = joblib.load(model_path)
    df = to_boxes(pd.read_csv(csv_path))
    by = list(df.groupby("filename"))[:limit]

    y_true, y_score = [], []
    cnt_true, cnt_pred = [], []

    from tqdm import tqdm
    for fname, g in tqdm(by, desc=f"Evaluating {split}"):
        img = cv2.imread(os.path.join(im_dir, fname))
        if img is None:
            continue

        gt = g[["xmin", "ymin", "xmax", "ymax"]].astype(int).values.tolist()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes, scores = detect_img(gray, clf, score_thr=score_thr)
        boxes, scores = nms(boxes, scores, nms_thr)

        before = len(boxes)

        filtered_boxes, filtered_scores = [], []
        for b, s in zip(boxes, scores):
            w = b[2] - b[0]
            h = b[3] - b[1]
            if w <= 0 or h <= 0:
                continue
            ratio = h / (w + 1e-9)
            if (w >= 30 and h >= 60 and 1.5 <= ratio <= 3.5):
                filtered_boxes.append(b)
                filtered_scores.append(s)
        boxes, scores = filtered_boxes, filtered_scores

        after = len(boxes)
        print(fname, "NMS:", before, "po filtrze:", after)

        used = set()
        for i, b in enumerate(boxes):
            best = -1
            best_iou = 0.0
            for j, gg in enumerate(gt):
                if j in used:
                    continue
                v = iou(b, gg)
                if v > best_iou:
                    best_iou = v
                    best = j

            if best != -1 and best_iou >= iou_thr:
                used.add(best)
                y_true.append(1)
                y_score.append(scores[i])
            else:
                y_true.append(0)
                y_score.append(scores[i])

        cnt_true.append(len(gt))
        cnt_pred.append(len(boxes))

    y_bin = [1 if s >= 0.5 else 0 for s in y_score]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_bin, average="binary", zero_division=0
    )
    mAP = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0
    mae = mean_absolute_error(cnt_true, cnt_pred)
    rmse = float(np.sqrt(((np.array(cnt_true) - np.array(cnt_pred)) ** 2).mean()))

    print(f"Split: {split}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  mAP: {mAP:.4f}")
    print(f"MAE (count): {mae:.3f}  RMSE (count): {rmse:.3f}")

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "mAP": float(mAP),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--score_thr", type=float, default=1.9)
    ap.add_argument("--nms_thr", type=float, default=0.2)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--model", type=str, default="hog_svm.joblib")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    main(
        args.split,
        iou_thr=args.iou_thr,
        nms_thr=args.nms_thr,
        score_thr=args.score_thr,
        model_path=args.model,
        limit=args.limit,
    )
