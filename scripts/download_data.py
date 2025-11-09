import kagglehub
import pandas as pd

root = kagglehub.dataset_download("adilshamim8/people-detection")

#train dataset download 
train_csv = f"{root}/train/train/_annotations.csv"
train_images = f"{root}/train/train"

df = pd.read_csv(train_csv)
print("Train shape:", df.shape)
print(df.head())

import os, shutil, glob

os.makedirs("data/train/images", exist_ok=True)

shutil.copy(train_csv, "data/train/annotations.csv")

for img in glob.glob(f"{train_images}/*.jpg"):
    shutil.copy(img, "data/train/images")

print("Gotowe dane treningowe")

#test dataset download 
test_csv = f"{root}/test/test/_annotations.csv"
test_images = f"{root}/test/test"

df = pd.read_csv(test_csv)
print("Test shape:", df.shape)
print(df.head())

import os, shutil, glob

os.makedirs("data/test/images", exist_ok=True)

shutil.copy(test_csv, "data/test/annotations.csv")

for img in glob.glob(f"{test_images}/*.jpg"):
    shutil.copy(img, "data/test/images")

print("Gotowe dane testowe")

#validation dataset download 
valid_csv = f"{root}/valid/valid/_annotations.csv"
valid_images = f"{root}/valid/valid"

df = pd.read_csv(valid_csv)
print("Validation shape:", df.shape)
print(df.head())

import os, shutil, glob

os.makedirs("data/valid/images", exist_ok=True)

shutil.copy(valid_csv, "data/valid/annotations.csv")

for img in glob.glob(f"{valid_images}/*.jpg"):
    shutil.copy(img, "data/valid/images")

print("Gotowe dane walidacyjne")
