import os, cv2, numpy as np
from glob import glob
from tqdm import tqdm
from skimage.feature import hog

IMG_DIR = "prepared/train/pos"
OUT_DIR = "preextracted/6x6-3x3-12/features_pos"
CHUNK_SIZE = 1000  # number of images per chunk

os.makedirs(OUT_DIR, exist_ok=True)

def extract_features_chunked(image_dir, out_dir, chunk_size):
    feats = []
    files = sorted(glob(os.path.join(image_dir, "*.png")))
    part = 0

    for i, p in enumerate(tqdm(files, desc=f"Extracting HOG from {image_dir}")):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue

        f = hog(im, pixels_per_cell=(6,6), cells_per_block=(3,3),
                orientations=12, block_norm="L2-Hys", feature_vector=True)
        feats.append(f)

        # save every chunk_size images
        if (i + 1) % chunk_size == 0:
            arr = np.array(feats, dtype=np.float32)
            np.save(os.path.join(out_dir, f"part_{part:03d}.npy"), arr)
            feats = []
            part += 1

    # save last partial batch
    if feats:
        arr = np.array(feats, dtype=np.float32)
        np.save(os.path.join(out_dir, f"part_{part:03d}.npy"), arr)

    print(f"✅ Saved HOG feature chunks to {out_dir}")

if __name__ == "__main__":
    extract_features_chunked(IMG_DIR, OUT_DIR, CHUNK_SIZE)
