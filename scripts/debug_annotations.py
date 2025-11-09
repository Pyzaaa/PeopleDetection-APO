import os, random, cv2, pandas as pd
import numpy as np

SPLIT = "train"
IM_DIR = f"data/{SPLIT}/images"
CSV = f"data/{SPLIT}/annotations.csv"

def normalize_boxes(df):
    cols = {c.lower(): c for c in df.columns}
    # wybór kolumn z klasą (opcjonalnie)
    class_col = None
    for k in ["class","label","category","name"]:
        if k in cols: class_col = cols[k]; break
    if class_col is not None:
        df = df[df[class_col].astype(str).str.lower().isin(["person","people","human"])]
    # mapowanie bbox
    if all(k in cols for k in ["xmin","ymin","xmax","ymax","filename"]):
        f = cols["filename"]; x1,y1,x2,y2 = [cols[k] for k in ["xmin","ymin","xmax","ymax"]]
        out = df[[f,x1,y1,x2,y2]].copy()
        out.columns = ["filename","xmin","ymin","xmax","ymax"]
    elif all(k in cols for k in ["x","y","width","height","filename"]):
        f = cols["filename"]; x,y,w,h = [cols[k] for k in ["x","y","width","height"]]
        tmp = df[[f,x,y,w,h]].copy()
        tmp["xmin"]=tmp[x]; tmp["ymin"]=tmp[y]
        tmp["xmax"]=tmp[x]+tmp[w]; tmp["ymax"]=tmp[y]+tmp[h]
        out = tmp[["filename","xmin","ymin","xmax","ymax"]]
    else:
        raise ValueError(f"Nieznany format CSV: {list(df.columns)}")
    return out

def denormalize_if_needed(boxes, img_w, img_h):
    # jeśli wygląda na znormalizowane (0-1), przeskaluj do pikseli
    if (boxes[["xmin","xmax"]].max().max() <= 1.01) and (boxes[["ymin","ymax"]].max().max() <= 1.01):
        boxes = boxes.copy()
        boxes["xmin"] *= img_w; boxes["xmax"] *= img_w
        boxes["ymin"] *= img_h; boxes["ymax"] *= img_h
    return boxes

df = pd.read_csv(CSV)
df = normalize_boxes(df)

# weź losowy plik i narysuj GT
fname = random.choice(df["filename"].unique().tolist())
path = os.path.join(IM_DIR, os.path.basename(fname))
img = cv2.imread(path)
assert img is not None, f"Nie znaleziono: {path}"

H,W = img.shape[:2]
g = df[df["filename"]==fname]
g = denormalize_if_needed(g, W, H)

vis = img.copy()
for _,r in g.iterrows():
    x1,y1,x2,y2 = map(int, [r.xmin, r.ymin, r.xmax, r.ymax])
    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.imwrite("debug_gt.jpg", vis)
print("Zapisano: debug_gt.jpg  (zielone = GT)")
