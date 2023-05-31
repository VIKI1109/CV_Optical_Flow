import cv2
import numpy as np
import cv2 as cv


class RectangleDrag:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv.setMouseCallback(win, self.onMouse)
        self.isDragStart = None
        self.isDragFinish = None

    def onMouse(self, event, x, y, flags, p):
        x, y = np.int16([x, y])
        if event == cv.EVENT_LBUTTONDOWN:
            self.isDragStart = (x, y)
            return
        if self.isDragStart:
            if flags & cv.EVENT_FLAG_LBUTTON:
                self.rec_selection(x, y)
            else:
                rect = self.isDragFinish
                self.isDragStart = None
                self.isDragFinish = None
                if rect:
                    self.callback(rect)

    def draw(self, vis):
        if not self.isDragFinish:
            return False
        x0, y0, x1, y1 = self.isDragFinish
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    def rec_selection(self, x, y):
        x_d, y_d = self.isDragStart
        x0, y0 = np.minimum([x_d, y_d], [x, y])
        x1, y1 = np.maximum([x_d, y_d], [x, y])
        self.isDragFinish = None
        if x1 - x0 > 0 and y1 - y0 > 0:
            self.isDragFinish = (x0, y0, x1, y1)


class App:
    def __init__(self, video_src, paused=False):
        self.cap = cv2.VideoCapture(video_src)
        _, self.frame = self.cap.read()
        cv.imshow('frame', self.frame)
        self.rectangle_drag = RectangleDrag('frame', self.select)
        self.trackers = []
        self.paused = paused

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
                for tracker in self.trackers:
                    tracker.update(frame_gray)

            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.draw_tracker_rectangle(vis)

            self.rectangle_drag.draw(vis)

            cv.imshow('frame', vis)
            ch = cv.waitKey(10)

            if ch == 27:
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('d'):
                self.trackers = []

    def select(self, rect):
        frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        tracker = Mosse(frame_gray, rect)  # initialize one mosse to each objects which is tracking
        self.trackers.append(tracker)


# random wrap the image to generate the training set
def random_warp_image(img):
    h, w = img.shape[:2]
    c, s = np.cos(np.random.uniform(-0.1, 0.1)), np.sin(np.random.uniform(-0.1, 0.1))
    W = np.array([[c + np.random.uniform(-0.1, 0.1), -s + np.random.uniform(-0.1, 0.1), 0],
                  [s + np.random.uniform(-0.1, 0.1), c + np.random.uniform(-0.1, 0.1), 0]])
    center_warp = np.array([[w / 2], [h / 2]])
    tmp = np.sum(W[:, :2], axis=1).reshape((2, 1))
    W[:, 2:] = center_warp - center_warp * tmp
    warped = cv2.warpAffine(img, W, (w, h), cv2.BORDER_REFLECT)
    return warped


def gaussian_response(sz, sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    center_x, center_y = w / 2, h / 2
    dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma ** 2)
    response = np.exp(-0.5 * dist)

    return response


class Mosse:

    def __init__(self, frame, rect):
        self.H2 = None
        self.H1 = None
        self.G = None
        self.fi = None
        self.size = None
        self.center = None
        self.x = None
        self.height = None
        self.y = None
        self.width = None
        self.learningRate = 0.125
        self.num_of_pretrain = 150
        self.sigma = 2.0
        self.initFilter(frame, rect)

    def initFilter(self, frame, rect):
        # obtain the rectangle attribute
        frame = frame.astype(np.float32) / 255

        x_left, y_up, x_right, y_down = rect
        # the Fourier-optimal size for a given dimension.
        self.width, self.height = map(cv.getOptimalDFTSize, [x_right - x_left, y_down - y_up])
        # the coordinate point for the left- up corner of the selected rectangle
        x1 = x_left
        y1 = y_up
        # initialize the center position
        self.x, self.y = x1 + 0.5 * (self.width - 1), y1 + 0.5 * (self.height - 1)
        self.center = (self.x, self.y)

        self.size = self.width, self.height
        w, h = int(round(self.width)), int(round(self.height))  # round()四舍五入
        self.fi = cv.getRectSubPix(frame, (w, h), self.center)
        # calculate the Gaussian response map g1 of the selected image fi.
        self.G = np.fft.fft2(gaussian_response((self.width, self.height), self.sigma))
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)

        # Pre-training to obtain the original Ai and Bi, which can be used to calculate the most suitable Gaussian
        # filter Hi.
        self.pre_training()


    def update(self, frame):
        H = self.H1 / self.H2  # 更新滤波器
        # calculate the filter

        # grey picture
        frame = frame.astype(np.float32) / 255.
        fi = cv.getRectSubPix(frame, (self.width, self.height), (self.x, self.y))
        # Preprocess the current frame fi and Calculate the Gaussian response Gi of fi
        Gi = H * np.fft.fft2(self.preprocess(fi))
        # The real gi is calculated by taking the inverse Fourier transform of Gi
        gi = np.real(np.fft.ifft2(Gi))
        curr = np.unravel_index(np.argmax(gi, axis=None), gi.shape)
        # the position of the maximum value in the gi is the position of the target in the new image frame
        y_add, x_add = curr[0] - (self.height / 2), curr[1] - (self.width / 2)
        x_center, y_center = self.center
        # Calculate the center of this target
        x_center += x_add
        y_center += y_add
        self.x = x_center
        self.y = y_center

        # update the center
        self.center = (x_center, y_center)
        # Use the new center to take a new frame
        fi = cv2.getRectSubPix(frame, (int(round(self.width)), int(round(self.height))),
                               (x_center, y_center))
        # Update the filter
        self.updateFilter(fi)

    # The image is multiplied by a cosine window so that the pixel values near the edges are gradually reduced to zero.
    def preprocess(self, img):
        height, width = img.shape
        img = np.log(img + 1)
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        window = self.window_hanning(height, width)
        img = img * window
        return img

    def window_hanning(self, height, width):
        column = np.hanning(width)
        row = np.hanning(height)
        column_x, row_x = np.meshgrid(column, row)
        win = column_x * row_x
        return win

    def draw_tracker_rectangle(self, vis):
        (x, y), (w, h) = self.center, self.size
        x1, y1, x2, y2 = int(x - 0.5 * w), int(y - 0.5 * h), int(x + 0.5 * w), int(y + 0.5 * h)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        cv.circle(vis, (int(x), int(y)), 10, (0, 0, 255), -1)

    def updateFilter(self, fi):
        fi = self.preprocess(fi)
        Fi = np.fft.fft2(fi)
        self.H1 = self.learningRate * (self.G * np.conj(Fi)) + (1 - self.learningRate) * self.H1
        self.H2 = self.learningRate * (Fi * np.conj(Fi)) + (1 - self.learningRate) * self.H2

    def pre_training(self):
        for _ in range(100):
            fi = random_warp_image(self.fi)
            Fi = np.fft.fft2(self.preprocess(fi))
            self.H1 += self.G * np.conj(Fi)
            self.H2 += Fi * np.conj(Fi)

if __name__ == '__main__':
    App("out.mov").run()
