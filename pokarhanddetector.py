from ultralytics import YOLO
import cv2
import cvzone
import math
import pokarhandfunction
img = cv2.imread("flush.png")

model = YOLO("playingCards.pt")
classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']


while True:
    # success, img = cap.read()
    results = model(img, stream = True)
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #for cv2
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #for cv zone
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(img, bbox)

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            cls = box.cls[0]

            cvzone.putTextRect(img, f'{conf} {classNames[int(cls)]}', (max(0, x1), max(0, y1-20)), scale=1, thickness=1)

            if conf > 0.5:
                hand.append(classNames[int(cls)])
        print(hand)
        hand = list(set(hand))
    if len(hand) == 5:
        results = pokarhandfunction.findPokerHand(hand)
        print(results)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
