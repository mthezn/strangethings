import cv2 
import numpy as np
import sys
import datetime
import imutils
import time
import cv2

def detect(frame,total):

    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray_video = cv2.resize(frame,(int (frame.shape[1]/2)),int(frame.shape[0]/2))
    #gray_video = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

    faces =  face_cascade.detectMultiScale(gray_video,1.1,4
)
    crop_img = []

    for (x, y, w, h )in faces:

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        crop_img = frame[y: y + h, x: x + w]
        roi_gray = gray_video[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        cv2.imwrite("face" + str(total) + ".jpg", crop_img)
        return 1

    return 0


video = cv2.VideoCapture('cut1.mp4')


face_cascade = cv2.CascadeClassifier('haarcascade_fullbody')
face_cascade.load("C:\\Users\\User\\Downloads\\opencv-master\\data\\haarcascades\\haarcascade_fullbody.xml")
total = 0
counter = 0
firstFrame = None
while(1):
    ret, frame = video.read()
    counter = counter + 1

    #cv2.waitKey(1000)
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if firstFrame is None:
        firstFrame = gray
        continue
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500 or cv2.contourArea(c) > 7000 :
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
    #if counter >= 10:

      #  if detect(frame,total):
       #     total = total + 1
        #    counter
         #   print(total)

        #counter = 0

    #frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

    #cv2.imshow("video", frame)
    #cv2.waitKey(20)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
      #  break
#print(str(len(faces)))
cv2.destroyAllWindows()
video.release()

