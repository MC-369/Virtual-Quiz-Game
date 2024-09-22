import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, detectionCon=0.5, maxHands=2):
        self.detectionCon = detectionCon
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                myHand = {}
                lmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    lmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                # Bounding box
                bbox = min(xList), min(yList), max(xList), max(yList)
                myHand["lmList"] = lmList
                myHand["bbox"] = bbox

                # Here we assume the first hand is "Right" and the second is "Left"
                # This is just a simple heuristic and might not be accurate.
                if flipType:
                    if len(allHands) == 0:
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = "Unknown"

                allHands.append(myHand)

                # Draw hand landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return allHands, img

    def findDistance(self, p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return length, [x1, y1, x2, y2]
