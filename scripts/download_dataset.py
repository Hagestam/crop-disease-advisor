# scripts/download_dataset.py
import urllib.request, zipfile, os

url = "https://storage.googleapis.com/plantvillage-dataset/color.tar"
print("Downloading PlantVillage...")
urllib.request.urlretrieve(url, "data/raw/plantvillage.tar")
import tarfile
with tarfile.open("data/raw/plantvillage.tar") as tar:
    tar.extractall("data/raw/")
print("Done.")