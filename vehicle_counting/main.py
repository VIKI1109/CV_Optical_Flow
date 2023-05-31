import cv2
import numpy as np


class Cars:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tracks = []
        self.done = False
        self.isSetDirection = '0'
        self.frame_number = 0
        self.max_frame_number = 5
        self.direction = None

    def get_centerx(self):
        return self.x

    def get_centery(self):
        return self.y

    def updateCentroid(self, x, y):
        self.frame_number = 0
        self.tracks.append([self.x, self.y])
        self.x = x
        self.y = y

    def remove(self):
        self.done = True

    def removeTheVehicle(self):
        return self.done

    def up(self, line):
        if len(self.tracks) >= 2 and self.isSetDirection == '0' and self.tracks[-1][1] < line <= self.tracks[-2][1]:
            self.direction = 'up'
            self.isSetDirection = '1'
            return True
        else:
            return False

    def down(self, line):
        if len(self.tracks) >= 2 and self.isSetDirection == '0' and self.tracks[-1][1] > line >= self.tracks[-2][1]:
            self.direction = 'down'
            self.isSetDirection = '1'
            return True
        else:
            return False

    #  if one car exist up to 5 frame , then remove the car from the array
    def add(self):
        self.frame_number += 1
        if self.frame_number > self.max_frame_number:
            self.done = True
        return True


count_number_for_up = 0
count_number_for_down = 0

cap = cv2.VideoCapture("count4.mov")

width = cap.get(3)
height = cap.get(4)

line = int(3 * (height / 5))

kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((11, 11), np.uint8)

point_left = [0, line]
point_right = [width, line]
points_array = np.array([point_left, point_right], np.int32)
points_array = points_array.reshape((-1, 1, 2))

# Create the background subtractor
mog_subtractor = cv2.createBackgroundSubtractorMOG2()

cars = []

while cap.isOpened():
    # read a frame
    ret, frame = cap.read()
    if not ret:
        break
    if ret:
        cv2.resize(frame, (640, 480))

    for i in cars:
        i.add()

    mog_mask = mog_subtractor.apply(frame)
    # Binary to shadow
    ret, dst = cv2.threshold(mog_mask, 150, 255, cv2.THRESH_BINARY)
    # Opening to remove noise
    mask = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel1)
    # Closing to join white region
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    for cnt in contours:

        area = cv2.contourArea(cnt)

        if (width * height) / 200 < area < 20000:
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3, 8)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            # the object is near the one which already detect before
            notContainInCars = True
            for i in cars:
                if abs(x - i.get_centerx()) <= w and abs(y - i.get_centery()) <= h:
                    notContainInCars = False
                    i.updateCentroid(cx, cy)  # Update the coordinates in the object and reset age
                    if i.up(line):
                        count_number_for_up += 1

                    elif i.down(line):
                        count_number_for_down += 1
                    break

                if i.removeTheVehicle():
                    index = cars.index(i)
                    cars.pop(index)
                    del i

            if notContainInCars:
                p = Cars(cx, cy)
                cars.append(p)

            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)


    frame = cv2.polylines(frame, [points_array], False, (0, 0, 255), thickness=2)

    # up = 'UP: ' + str(count_number_for_up)
    # cv2.putText(frame, up, (400, 50), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(frame, up, (400, 50), font, 2, (0, 0, 255), 1, cv2.LINE_AA)

    down = 'DOWN: ' + str(count_number_for_down)
    cv2.putText(frame, down, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, down, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()



