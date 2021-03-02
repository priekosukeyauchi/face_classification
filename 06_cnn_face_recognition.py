#python 05_cnn_face_recognition.py model/final_model.h5
import tensorflow.keras
import numpy as np
import cv2
import os, sys

cv_width, cv_height = 64, 64
minN = 15
#difinition image size
img_width, img_height = 64, 64

#directory for train_data_set
train_data_dir = "dataset/03_face_inflated"
#directory for test_dataset
test_data_dir = "dataset/05_test"
#directory for face_detection
rec_data_dir = "dataset/06_rec"

classes = os.listdir(train_data_dir)

cascade_xml = "haarcascade_frontalface_default.xml"


def detect_face (image, model):
    #convert gray scale
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_xml)
    
    #detect face
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    #
    if len(face_list) > 0:
        for rect in face_list:
            print(face_list)
            face_img = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                print("too small")
                continue
            #definision of face_img and size
            face_img = cv2.resize(face_img, (img_width, img_height))
            #conversion BGR->RGB, float_type for keras
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            #recognition face_img
            name = predict_who(face_img, model)
            

            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness = 3)

            x, y, width, height = rect
            cv2.putText(image, name, (x, y + height+60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)

    else:
        print("No face")
    return image

def predict_who(x, model):
    #画像データをテンソル整形
    x = np.expand_dims(x, axis=0)
    #データ正規化
    x = x/255

    pred = model.predict(x)[0]

    top = 2
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    print(result)
    print('=======================================')

    return result[0][0]



def main():
    savefile = "model/final_model.h5"

    model = tensorflow.keras.models.load_model(savefile)
    
  
    test_imagelist = os.listdir(test_data_dir)
    for test_image in test_imagelist:
        file_name = os.path.join(test_data_dir, test_image)
        print(file_name)

        image = cv2.imread(file_name)
        if image is None:
            print("Not open:", file_name)
            continue

        rec_image = detect_face(image, model)

        rec_file_name = os.path.join(rec_data_dir, "rec-" + test_image)
        cv2.imwrite(rec_file_name, rec_image)


if __name__ == "__main__":
    main()
        
