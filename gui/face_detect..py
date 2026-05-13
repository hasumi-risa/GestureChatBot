import cv2 as cv

HAAR_FILE = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"
# HAAR_FILE = "opencv/data/haarcascades/haarcascade_eye.xml"
cascade = cv.CascadeClassifier(HAAR_FILE)

cap = cv.VideoCapture(0)


import time
ts = time.time()
none_count = 0
somenone_ct = 0
while(True):
    ret, frame = cap.read()

    face = cascade.detectMultiScale(frame)
    
    if len(face):
        tf = time.time()
        if tf - ts > 1.0 :
            somenone_ct += 1
            if somenone_ct == 1:    
                print("音声認識スタート") 
                none_count = 0
    else:
        ts = time.time()
        none_count += 1
        if none_count == 100:
            print("誰もいない")
            none_count = 0
            somenone_ct = 0

    for x, y, w, h in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()