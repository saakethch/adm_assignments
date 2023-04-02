from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from datetime import datetime
import os
if __name__ == '__main__':
    fe = FeatureExtractor()
    root = "./static/img"
    for root, dirs, files in os.walk(root):
        for file_name in files:
            img_path = os.path.join(root, file_name)
            print(img_path)
            feature = fe.extract(img=Image.open(img_path))
            feature_path = Path("./static/feature/"+  file_name+ ".npy") 
            print(feature_path)
            np.save(feature_path, feature)
