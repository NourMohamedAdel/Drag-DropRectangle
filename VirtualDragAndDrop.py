import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand detector
detector = HandDetector(detectionCon=0.8)
colorR = (255,0,255) 

# Initial rectangle position and size
cx, cy , w, h= 100,100,200,200

# Class to handle draggable rectangles
class DragRect():
    def __init__(self, posCenter, size=[200,200]):
        self.posCenter = posCenter
        self.size = size

    def update(self,cursor):
        cx,cy = self.posCenter
        w,h = self.size
        # if the index finger tip is in the rectangle region
        if cx - w//2 < cursor[0] < cx + w//2 and cy - h//2 < cursor[1] < cy + h//2:
            self.posCenter = cursor[:2]

# Create multiple draggable rectangles
rectList = []        
for x in range(5):
    rectList.append(DragRect([x*250+150,150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hands[0]['type'] = ''
        lmList = hands[0]['lmList']

        if lmList:

            point1 = lmList[8][:2]   # Index finger tip
            point2 = lmList[12][:2]  # Middle finger tip
            l, _, _ = detector.findDistance(point1,point2,img)
            print(l)
            if l<40:

                cursor = lmList[8]  # Index finger tip landmark
                # call the update here
                for rect in rectList:
                    rect.update(cursor)

   
    ## Draw solid
    # for rect in rectList:
    #     cx,cy = rect.posCenter
    #     w,h = rect.size           
    #     cv2.rectangle(img, (cx-w//2,cy-h//2), (cx+w//2,cy+h//2), colorR, cv2.FILLED)
    #     cvzone.cornerRect(img, (cx-w//2,cy-h//2, w, h), 20, rt=0)

    # Draw with transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx,cy = rect.posCenter
        w,h = rect.size           
        cv2.rectangle(imgNew, (cx-w//2,cy-h//2), (cx+w//2,cy+h//2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx-w//2,cy-h//2, w, h), 20, rt=0)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

