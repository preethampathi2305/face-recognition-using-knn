import cv2
import numpy as np
import os

#KNN:
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(ds,querypoint,k=5):
    vals=[]
    m=ds.shape[0]
    
    for i in range(m):
        x=ds[i,:-1]
        y=ds[i,-1]
        d=dist(querypoint,x)
        vals.append([d,y])
        
    vals=sorted(vals,key=lambda x:x[0])
    vals=vals[:k]
    vals=np.array(vals)
    new_vals=np.unique(vals[:,-1],return_counts=True)
    index=new_vals[1].argmax()
    fin=new_vals[0][index]
    return fin

#cv2 code:

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_list=[]
dt="./data/"

face_data=[]
label=[]

class_id=0
names=[]

for f in os.listdir(dt):
    if f.endswith('.npy'):
        temp=np.load(dt+"\\"+f)
        face_data.append(temp)
        names.append(f[:-4])
        col=class_id*np.ones((temp.shape[0],))
        class_id+=1
        label.append(col)
face_data=np.concatenate(face_data,axis=0)
label=np.concatenate(label,axis=0).reshape((-1,1))
trainset=np.concatenate((face_data,label),axis=1)



while True:
    ret,frame=cap.read()
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for i in faces:
        x,y,h,w=i
        offset=20
        croparea=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        croparea=cv2.resize(croparea,(100,100))
        out=knn(trainset,croparea.flatten())
        predicted_name=names[int(out)]
        cv2.putText(frame,predicted_name,(x+20,y-10),cv2.FONT_ITALIC,1,(255,255,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Video Frame",frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()