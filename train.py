import cv2
import numpy as np
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_list=[]
dt="./data/"
filename=input("Enter the name of the person:")

while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for i in faces:
        x,y,w,h=i
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Press C to capture",frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
    elif key_pressed==ord('c'):
        offset=20
        for i in faces:
            x,y,w,h=i
            croparea=frame[y-offset:y+h+offset,x-offset:x+w+offset]
            cv2.imshow("Captured",croparea)
            croparea=cv2.resize(croparea,(100,100))
            face_list.append(croparea)
            
face_list=np.asarray(face_list)
face_list=face_list.reshape((face_list.shape[0],-1))
print(face_list.shape)
np.save(dt+"/"+filename+'.npy',face_list)

cap.release()
cv2.destroyAllWindows()