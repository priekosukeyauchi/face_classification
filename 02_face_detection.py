#googleimagesdownload -k "" -l 20


import cv2
import os
import shutil

org_dir = "dataset/01_org/"
face_dir = "dataset/02_face/"

cascade_xml = "haarcascade_frontalface_default.xml"


def main():
    name_list = [filename for filename in os.listdir(org_dir) if not filename.startswith(".")]
    print(name_list)
    

    for name in name_list:
        name.replace(" ", "_")
        org_char_dir = org_dir + name + "/"
        print(org_char_dir)

        face_char_dir = face_dir + name + "/"
        os.makedirs(face_char_dir, exist_ok=True)

        print(len(face_char_dir))

        detect_face(org_char_dir, face_char_dir)

def detect_face(org_char_dir, face_char_dir):
    image_list = os.listdir(org_char_dir)

    for image_file in image_list:
        
        org_image = cv2.imread(org_char_dir + image_file)

        if org_image is None:
            print("Not open:", image_file)
            continue

        #convert gray_scale
        img_gs = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
        
        #detect_face
        cascade = cv2.CascadeClassifier(cascade_xml)

        for i_mn in range(1, 7, 1):
            face_list = cascade.detectMultiScale(img_gs, scaleFactor=1.1, minNeighbors=i_mn, minSize=(200, 200))
            #if more than one_face detected, get image (64*64)
            if len(face_list ) > 0:
                for rect in face_list:
                    image = org_image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
                    if image.shape[0] < 64 or image.shape[1] < 64:
                        continue
                    face_image = cv2.resize(image, (64, 64))

            else:
                continue
            

            #save face_image
            face_file_name = os.path.join(face_char_dir, "face-" + image_file)
            cv2.imwrite(str(face_file_name), face_image)

if __name__ == "__main__":
        main()






 