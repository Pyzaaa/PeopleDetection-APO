import cv2, joblib
from skimage.feature import hog
from utils import nms

def pyramid(gray, scale=1.25, min_size=(128,128)):
    yield gray, 1.0
    while True:
        h,w = gray.shape[:2]
        w2,h2 = int(w/scale), int(h/scale)
        if h2<min_size[1] or w2<min_size[0]: break
        gray = cv2.resize(gray, (w2,h2))
        yield gray, (1.0/scale)

def sliding(gray, step=8, win=(64,128)):
    H,W = gray.shape[:2]
    for y in range(0, H-win[1]+1, step):
        for x in range(0, W-win[0]+1, step):
            yield x,y,gray[y:y+win[1], x:x+win[0]]

def detect_image(img_path, model="hog_svm.joblib", score_thr=1.9):
    clf = joblib.load(model)
    img = cv2.imread(img_path); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes, scores = [], []
    inv=1.0
    for scaled, inv in pyramid(gray):
        for x,y,patch in sliding(scaled):
            f = hog(patch, pixels_per_cell=(6,6), cells_per_block=(3,3),
                    orientations=12, block_norm="L2-Hys", feature_vector=True)
            s = clf.decision_function([f])[0]
            if s >= score_thr:
                sx,sy = int(x/inv), int(y/inv)
                ex,ey = int((x+64)/inv), int((y+128)/inv)
                boxes.append([sx,sy,ex,ey]); scores.append(float(s))
    boxes, scores = nms(boxes, scores, 0.2)
    for b in boxes:
        cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (0,255,0), 2)
    out="detections_preview.jpg"
    cv2.imwrite(out, img)
    print("Zapisano:", out, "Wykryto:", len(boxes))

if __name__=="__main__":
    detect_image("data/valid/images/group-of-people-in-a-meeting-1367272-2_jpg.rf.9a2d1e6086b3691f11d52167cad402fa.jpg")
