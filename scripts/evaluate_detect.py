import os, cv2, pandas as pd, numpy as np, joblib, argparse
from skimage.feature import hog
from utils import nms
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, mean_absolute_error
from math import sqrt

def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9
    return inter/ua

def to_boxes(df):
    cols = {c.lower(): c for c in df.columns}
    if all(k in cols for k in ["xmin","ymin","xmax","ymax","filename"]):
        x1,y1,x2,y2 = [cols[k] for k in ["xmin","ymin","xmax","ymax"]]
        fcol = cols["filename"]
        boxes = df[[fcol,x1,y1,x2,y2]].copy()
        boxes.columns = ["filename","xmin","ymin","xmax","ymax"]
        return boxes
    elif all(k in cols for k in ["x","y","width","height","filename"]):
        fcol = cols["filename"]; x,y,w,h=[cols[k] for k in ["x","y","width","height"]]
        tmp = df[[fcol,x,y,w,h]].copy()
        tmp["xmin"]=tmp[x]; tmp["ymin"]=tmp[y]
        tmp["xmax"]=tmp[x]+tmp[w]; tmp["ymax"]=tmp[y]+tmp[h]
        return tmp[[fcol,"xmin","ymin","xmax","ymax"]]
    else:
        raise ValueError("Nieznany format CSV")

def detect_img(gray, clf):
    boxes, scores = [], []
    cur=gray.copy(); inv=1.0; scale=1.25
    while cur.shape[0]>=128 and cur.shape[1]>=64:
        for y in range(0,cur.shape[0]-128+1,8):
            for x in range(0,cur.shape[1]-64+1,8):
                p = cur[y:y+128, x:x+64]
                f = hog(p, pixels_per_cell=(8,8), cells_per_block=(2,2),
                        orientations=9, block_norm="L2-Hys", feature_vector=True)
                s = clf.decision_function([f])[0]
                sx,sy=int(x/inv),int(y/inv); ex,ey=int((x+64)/inv),int((y+128)/inv)
                boxes.append([sx,sy,ex,ey]); scores.append(float(s))
        cur = cv2.resize(cur, (int(cur.shape[1]/scale), int(cur.shape[0]/scale)))
        inv/=scale
    return boxes, scores

def main(split, iou_thr=0.5, nms_thr=0.4):
    im_dir=f"data/{split}/images"
    csv_path=f"data/{split}/annotations.csv"
    clf = joblib.load("hog_svm_test_metrics_6x6.joblib")
    df = to_boxes(pd.read_csv(csv_path))
    by = df.groupby("filename")
    by = list(by)[:10]  # tylko pierwsze 10 zdjęć

    y_true, y_score = [], []
    cnt_true, cnt_pred = [], []

    from tqdm import tqdm
    for fname, g in tqdm(by, desc=f"Evaluating {split}"):
        img = cv2.imread(os.path.join(im_dir, fname))
        if img is None: continue
        gt = g[["xmin","ymin","xmax","ymax"]].astype(int).values.tolist()
        boxes, scores = detect_img(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), clf)
        boxes, scores = nms(boxes, scores, nms_thr)

        used=set()
        for i,b in enumerate(boxes):
            best=-1; best_iou=0
            for j,g in enumerate(gt):
                if j in used: continue
                v=iou(b,g)
                if v>best_iou: best_iou=v; best=j
            if best!=-1 and best_iou>=iou_thr:
                used.add(best); y_true.append(1); y_score.append(scores[i])
            else:
                y_true.append(0); y_score.append(scores[i])

        cnt_true.append(len(gt)); cnt_pred.append(len(boxes))

    # punktowy PR/F1 przy progu 0.5 oraz AP (mAP dla klasy 1)
    y_bin = [1 if s>=0.5 else 0 for s in y_score]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_bin, average="binary", zero_division=0)
    mAP = average_precision_score(y_true, y_score)
    mae = mean_absolute_error(cnt_true, cnt_pred)
    rmse = np.sqrt(((np.array(cnt_true)-np.array(cnt_pred))**2).mean())

    print(f"Split: {split}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  mAP: {mAP:.4f}")
    print(f"MAE (count): {mae:.3f}  RMSE (count): {rmse:.3f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid","test"])
    args=ap.parse_args()
    main(args.split)
