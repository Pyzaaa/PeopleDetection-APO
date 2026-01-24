import os, cv2, numpy as np
from glob import glob
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

POS_DIR = "prepared/train/pos"
NEG_DIR = "prepared/train/neg"
MODEL_PATH = "hog_svm2.joblib"

def feats(d):
    X=[]
    for p in tqdm(sorted(glob(os.path.join(d,"*.png"))), desc=f"HOG {d}"):

        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        f = hog(im, pixels_per_cell=(8,8), cells_per_block=(2,2),
                orientations=9, block_norm="L2-Hys", feature_vector=True)
        X.append(f)
    return np.array(X, dtype=np.float32)

X_pos = feats(POS_DIR)
X_neg = feats(NEG_DIR)
X = np.vstack([X_pos, X_neg])
y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

rng = np.random.default_rng(123)
perm = rng.permutation(len(X))
split = int(0.8*len(X))
tr, te = perm[:split], perm[split:]

clf = LinearSVC(class_weight="balanced", max_iter=5000, random_state=123)
clf.fit(X[tr], y[tr])

print(classification_report(y[te], clf.predict(X[te]), digits=4))
joblib.dump(clf, MODEL_PATH)
print("Saved:", MODEL_PATH)
