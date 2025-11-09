import os, cv2, pandas as pd, numpy as np
from tqdm import tqdm
import argparse

OUT_SIZE = (64,128)

def to_boxes(df):
    cols = {c.lower(): c for c in df.columns}
    if all(k in cols for k in ["xmin","ymin","xmax","ymax","filename"]):
        x1,y1,x2,y2 = [cols[k] for k in ["xmin","ymin","xmax","ymax"]]
        fcol = cols["filename"]
        boxes = df[[fcol,x1,y1,x2,y2]].copy()
        boxes.columns = ["filename","xmin","ymin","xmax","ymax"]
        return boxes
    elif all(k in cols for k in ["x","y","width","height","filename"]):
        # zamiana (x,y,w,h) -> (xmin,ymin,xmax,ymax)
        fcol = cols["filename"]
        x,y,w,h = [cols[k] for k in ["x","y","width","height"]]
        tmp = df[[fcol,x,y,w,h]].copy()
        tmp["xmin"] = tmp[x]
        tmp["ymin"] = tmp[y]
        tmp["xmax"] = tmp[x] + tmp[w]
        tmp["ymax"] = tmp[y] + tmp[h]
        boxes = tmp[[fcol,"xmin","ymin","xmax","ymax"]]
        return boxes
    else:
        raise ValueError(f"Nieznany format CSV. Kolumny: {list(df.columns)}")

def iou(a,b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9
    return inter/ua

def main(split):
    im_dir = f"data/{split}/images"
    csv_path = f"data/{split}/annotations.csv"
    pos_dir = f"prepared/{split}/pos"
    neg_dir = f"prepared/{split}/neg"
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = to_boxes(df)
    by_img = df.groupby("filename")

    pid=nid=0
    rng = np.random.default_rng(123)

    for fname, g in tqdm(by_img, desc=f"Preparing {split} patches"):
        path = os.path.join(im_dir, fname)
        img = cv2.imread(path)
        if img is None: 
            continue
        H,W = img.shape[:2]
        gts = g[["xmin","ymin","xmax","ymax"]].astype(int).values

        #pozytywy
        for x1,y1,x2,y2 in gts:
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W,x2), min(H,y2)
            if x2-x1>=16 and y2-y1>=32:
                crop = cv2.resize(img[y1:y2, x1:x2], OUT_SIZE, cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(pos_dir, f"pos_{pid}.png"), crop)
                pid += 1

        #negatywy
        target = min(20, 5*len(gts))
        trials=0
        while target>0 and trials<400:
            trials+=1
            ww,hh = OUT_SIZE
            x1 = int(rng.integers(0, max(1,W-ww)))
            y1 = int(rng.integers(0, max(1,H-hh)))
            x2,y2 = x1+ww, y1+hh
            if all(iou((x1,y1,x2,y2), gt)<0.2 for gt in gts):
                cv2.imwrite(os.path.join(neg_dir, f"neg_{nid}.png"), img[y1:y2, x1:x2])
                nid += 1
                target -= 1

    print(f"Done {split}. Pos={pid} Neg={nid}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train","valid","test"])
    args = ap.parse_args()
    main(args.split)
