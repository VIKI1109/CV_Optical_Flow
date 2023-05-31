import cv2
import numpy as np

from LK_pymaid.lk import lucas_kanade
from lk_pyr import lucas_kanade_pyramid

feature_params = dict(maxCorners=100,
                      qualityLevel=0.1,
                      minDistance=5,
                      blockSize=7)

lk_params = dict(winSize=(11, 11),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class App:

    def __init__(self, cap):
        self.previourGreyFrame = None
        self.track_len = 15
        self.goodFeatureUpdateTime = 5
        self.tracks = []
        self.number_of_frame = 0
        self.cap = cap

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                current_frame = frame.copy()

                if len(self.tracks) > 0:
                    firstImage, secondImage = self.previourGreyFrame, frame_gray

                    original_position = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                    # Then check whether the feature points have been tracked successfully. The second image is used
                    # as the initial frame, the first image is used as the next frame, and the transformed feature
                    # points are passed into the LK algorithm as the feature points of the initial frame to obtain
                    # the optical flow value. Then the new position of the feature points on the first image is
                    # found. By comparing the original feature point positions with these positions, the feature
                    # point has been tracked if the difference between them is less than one.

                    (u, v) = lucas_kanade_pyramid(firstImage, secondImage, original_position, 5, 3, 4)
                    flow = np.array(list(zip(list(u), list(v))))
                    original_position = np.array(np.reshape(original_position, (-1, 2)))
                    flow = np.array(np.reshape(flow, (-1, 2)))

                    position_after_transformed = np.array(original_position + flow)

                    (u1, v1) = lucas_kanade_pyramid(secondImage, firstImage, position_after_transformed, 5, 3, 4)
                    flow1 = np.array(list(zip(list(u1), list(v1))))
                    p11 = np.reshape(position_after_transformed, (-1, 2))
                    flow1 = np.reshape(flow1, (-1, 2))
                    back_trace = np.array(p11 + flow1)
                    original_position = np.array(original_position)

                    d = abs(original_position - back_trace).reshape(-1, 2).max(-1)
                    cornerIsDetected = d < 1
                    new_tracks = []

                    for track, (x, y), isCornerdetected in zip(self.tracks, position_after_transformed,
                                                               cornerIsDetected):
                        if not isCornerdetected:
                            continue
                        track.append((np.float32(x), np.float32(y)))
                        if len(track) > self.track_len:
                            del track[0]

                        new_tracks.append(track)
                        cv2.circle(current_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    # If the feature point is tracked, then the position of the transformed feature point is recorded
                    # in the track array. A line is drawn from the position of the previous frame's feature point to
                    # the position of the current frame's feature point as a representation of the optical flow. draw
                    # lines for all the position of the tracking point
                    cv2.polylines(current_frame, [np.int64(track) for track in self.tracks], False, (0, 255, 0), 1)

                if self.number_of_frame % self.goodFeatureUpdateTime == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int64(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 2, 0, -1)

                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float64(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

                self.number_of_frame += 1
                self.previourGreyFrame = frame_gray
                cv2.imshow('lk_pyramid', current_frame)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    import sys

    video_src = "out.mov"
    cap = cv2.VideoCapture(video_src)
    App(cap).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
