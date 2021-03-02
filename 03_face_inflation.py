import os
import cv2
import glob
from scipy import ndimage
import numpy as np

org_dir = "dataset/01_org/"
face_dir = "dataset/02_face/"

namelist = [filename for filename in os.listdir(face_dir) if not filename.startswith(".")]
print(namelist)


for name in namelist:
    in_dir = face_dir + name
    in_dir_file = face_dir + name + "/*"
    out_dir = "dataset/03_face_inflated/" + name + "/"
    os.makedirs(out_dir, exist_ok=True)
    in_jpg = glob.glob(in_dir_file)
    print(in_jpg)
    img_file_list = os.listdir(in_dir)
    for i in range(len(in_jpg)):
        img = cv2.imread(str(in_jpg[i]))
        for ang in [-10, 0, 10]:
            img_rot = ndimage.rotate(img, ang)
            img_rot = cv2.resize(img_rot, (64, 64))
            fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + ".jpg")
            cv2.imwrite(str(fileName), img_rot)

            img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
            fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + "thr.jpg")
            cv2.imwrite(str(fileName), img_thr)

            img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"filter.jpg")
            cv2.imwrite(str(fileName), img_filter)

            min_table = 50
            max_table = 205
            diff_table = max_table - min_table
            LUT_HC = np.arange(256, dtype = 'uint8' )
            LUT_LC = np.arange(256, dtype = 'uint8' )

            # ハイコントラストLUT作成
            for r in range(0, min_table):
                LUT_HC[r] = 0
            for r in range(min_table, max_table):
                LUT_HC[r] = 255 * (r - min_table) / diff_table
            for r in range(max_table, 255):
                LUT_HC[r] = 255

            # ローコントラストLUT作成
            for r in range(256):
                LUT_LC[r] = min_table + r * (diff_table) / 255

            # 変換
            
            low_cont_img = cv2.LUT(img_rot, LUT_LC)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"low_lut.jpg")
            cv2.imwrite(str(fileName), low_cont_img)

            high_cont_img = cv2.LUT(img_rot, LUT_HC)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"high_lut.jpg")
            cv2.imwrite(str(fileName), high_cont_img)


            row,col,ch = img_rot.shape
            s_vs_p = 0.5
            amount = 0.004
            sp_img = img_rot.copy()

            # 塩モード
            num_salt = np.ceil(amount * img_rot.size * s_vs_p)
            coords = [np.random.randint(0, i-1 , int(num_salt)) for i in img_rot.shape]
            sp_img[coords[:-1]] = (255,255,255)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"salt.jpg")
            cv2.imwrite(str(fileName), sp_img)

            # 胡椒モード
            num_pepper = np.ceil(amount* img_rot.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i-1 , int(num_pepper)) for i in img_rot.shape]
            sp_img[coords[:-1]] = (0,0,0)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"pepper.jpg")
            cv2.imwrite(str(fileName), sp_img)

            
            hflip_img = cv2.flip(img_rot, 1)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"hflip.jpg")
            cv2.imwrite(str(fileName), hflip_img)
            vflip_img = cv2.flip(img_rot, 0)
            fileName = os.path.join(out_dir, str(i)+"_"+str(ang)+"vflip.jpg")
            cv2.imwrite(str(fileName), vflip_img)

    print(name + ":" + str(len(out_dir))+"images")




print("完了")


