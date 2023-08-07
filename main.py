import cv2
import pickle, numpy as np
from random import randrange
import os

face_cap = cv2.CascadeClassifier(
    "C:/Users/arsla/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

video_cap = cv2.VideoCapture(0)
face_data=[]
i=0

name=input("Enter your name: ")
while True:

    ret, video = video_cap.read()
    #col        = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cap.detectMultiScale(
        frame_gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        crop_img=video[y:y+h,x:x+w,:]
        resize_img=cv2.resize(crop_img,(50,50))
        if len(face_data)<=50 and i%10==0:
            face_data.append(resize_img)
        i=i+1
        cv2.putText(video,str(len(face_data)),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
        frame = cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 4)

        faceROI = frame_gray[y:y + h, x:x + w]



    cv2.imshow("Live_Video", video)
    if cv2.waitKey(1) == ord("a") or len(face_data)==50:
        break
video_cap.release()

####################

faces_data=np.asarray(face_data)
faces_data=faces_data.reshape(50,-1)

#Checking if any file is available or not
if  'names.pkl' not in os.listdir('data/'):
    names= [name] *50
    with open('data/names.pkl','wb') as f :
        pickle.dump(names,f)
else:
    with open('data/names.pkl','rb') as f :
        names=pickle.load(f)
    names=names[names]*50
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
# For faces
if  'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl','wb') as f :
        pickle.dump(faces_data,f)
else:
    with open('data/faces_data.pkl','rb') as f :
        faces=pickle.load(f)
    faces=np.append(faces,faces_data,axis=0)
    with open('data/faces_data.pkl','wb') as f:
        pickle.dump(names,f)

