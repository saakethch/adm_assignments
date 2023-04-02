from pathlib import Path
import os
import shutil
import shortuuid

root = "./images/MEN/"
dst_path = "./images/img_data/"
for root, dirs, files in os.walk(root):
    for file_name in files:
        img_path = os.path.join(root, file_name)
        shutil.copyfile(img_path, dst_path + shortuuid.ShortUUID().random(length=10) + file_name)

