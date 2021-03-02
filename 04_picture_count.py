import os
import shutil
import glob
import random

face_dir = "dataset/03_face_inflated/"
name_list = [filename for filename in os.listdir(face_dir) if not filename.startswith(".")]
print(name_list)

for name in name_list:
    in_dir = face_dir + name + "/*"
    in_jpg = glob.glob(in_dir)
    img_file_name_list = os.listdir(face_dir + name + "/")
    random.shuffle(in_jpg)
    os.makedirs("dataset/04_validation/" + name, exist_ok=True)
    for t in range(len(in_jpg)//5):
        shutil.move(str(in_jpg[t]), "dataset/04_validation/" + name)

