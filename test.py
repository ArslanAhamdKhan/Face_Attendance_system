from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch


face_cap = cv2.CascadeClassifier(
    "C:/Users/arsla/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# For Speaking attendance
def speak(speech):
    speak= Dispatch(("SAPI.SpVoice"))
    speak.Speak(speech)
#For put text stationary
img = np.zeros((640,480,3), np.uint8)


video_cap = cv2.VideoCapture(0)

# Loading the face and names (label)
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl','rb') as f :
    FACES=pickle.load(f)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)


COL_NAMES=['NAME','TIME']
while True:

    ret, frame = video_cap.read()
    #col        = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cap.detectMultiScale(
        frame_gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    cv2.putText(frame, "Press 'o' for attendance ", (120, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 153, 0), 2, cv2.LINE_4)
    for (x, y, w, h) in faces:
        crop_img=frame[y:y+h,x:x+w,:]
        resize_img=cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output=knn.predict(resize_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")

        exist=os.path.isfile("Attendance/Attendance"+ date+ ".csv")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame, (x, y), (x + w,y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x + w,y), (50,50,255), -1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        attendance=[str(output[0]),str(timestamp)]
    cv2.imshow("Live_Video", frame)
    k=cv2.waitKey(1)
    if k==ord("a"):
        break
    else:
        if k == ord("o"):
            speak("Attendance Taken")
            time.sleep(1)
            if exist:
                with open("Attendance/Attendance" + date + ".csv", "+a") as f:
                    writer = csv.writer(f)
                    writer.writerow(attendance)
                f.close()
            else:
                with open("Attendance/Attendance" + date + ".csv", "+a") as f:
                    writer = csv.writer(f)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
                f.close()
        elif k==ord("a"):
            break

video_cap.release()
cv2.destroyAllWindows()

####################

