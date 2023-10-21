import wget

# Download the data from the source
source = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"

# target directory
target_dir = "data/raw/"

# download the data
wget.download(source, target_dir)

# unzip the data
import zipfile

with zipfile.ZipFile(target_dir + "filtered_paranmt.zip", "r") as zip_ref:
    zip_ref.extractall(target_dir)

# remove the zip file
import os

os.remove(target_dir + "filtered_paranmt.zip")
