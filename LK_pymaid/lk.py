import scipy.ndimage
import scipy.signal
from numpy import linalg
from pylab import *


def lucas_kanade(firstImage, secondImage, features, N):
    firstImage = np.array(firstImage)
    secondImage = np.array(secondImage)

    firstImage = firstImage / 255.
    secondImage = secondImage / 255.
    window = int(N // 2)

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[0.5, 0, -0.5]])
    kernel_y = np.array([[0.5], [0], [-0.5]])
    kernel_t = np.array([[-1]])

    Ix = scipy.ndimage.convolve(input=firstImage, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=firstImage, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=secondImage, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(
        input=firstImage, weights=-kernel_t, mode="nearest"
    )

    feature_change_u_array = [0 for i in range(len(features))]
    feature_change_v_array = [0 for i in range(len(features))]

    index = 0

    A = np.zeros((2, 2))
    b = np.zeros((2, 1))

    for l in features:
        i, j = l.ravel()
        i = int(i)
        j = int(j)

        Ix_win = Ix[
                      j - window: j + window + 1,
                      i - window: i + window + 1,
                      ].flatten()
        Iy_win = Iy[
                      j - window: j + window + 1,
                      i - window: i + window + 1,
                      ].flatten()
        It_win = It[
                      j - window: j + window + 1,
                      i - window: i + window + 1,
                      ].flatten()

        A[0][0] += np.dot(Ix_win, Ix_win)
        A[0][1] += np.dot(Ix_win, Iy_win)
        A[1][0] += np.dot(Ix_win, Iy_win)
        A[1][1] += np.dot(Iy_win, Iy_win)
        b[0][0] += np.dot(Ix_win, It_win)
        b[1][0] += np.dot(Iy_win, It_win)

        Ainv = np.linalg.pinv(A)
        (u, v) = np.dot(Ainv, b)

        feature_change_u_array[index] = u
        feature_change_v_array[index] = v
        index = index + 1

    return feature_change_u_array, feature_change_v_array


def lucas_kanade_tradition(firstImage, secondImage, N, image_ind=None, dataset=None, tau=1e-3):

    firstImage = np.array(firstImage)
    secondImage = np.array(secondImage)

    firstImage = firstImage / 255
    secondImage = secondImage / 255
    image_shape = firstImage.shape
    window = int(N // 2)

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[0.5, 0, -0.5]])
    kernel_y = np.array([[0.5], [0], [-0.5]])
    kernel_t = np.array([[-1]])

    Ix = scipy.ndimage.convolve(input=firstImage, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=firstImage, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(input=secondImage, weights=kernel_t, mode="nearest") + scipy.ndimage.convolve(
        input=firstImage, weights=-kernel_t, mode="nearest"
    )

    A = np.zeros((2, 2))
    b = np.zeros((2, 1))

    u = np.zeros(image_shape)
    v = np.zeros(image_shape)

    # Find Lucas Kanade OF for a block N x N with least squares solution
    for j in range(window, image_shape[0] - window):
        for i in range(window, image_shape[1] - window):
            Ix_win = Ix[
                          j - window: j + window + 1,
                          i - window: i + window + 1,
                          ].flatten()
            Iy_win = Iy[
                          j - window: j + window + 1,
                          i - window: i + window + 1,
                          ].flatten()
            It_win = It[
                          j - window: j + window + 1,
                          i - window: i + window + 1,
                          ].flatten()

            A[0][0] += np.dot(Ix_win, Ix_win)
            A[0][1] += np.dot(Ix_win, Iy_win)
            A[1][0] += np.dot(Ix_win, Iy_win)
            A[1][1] += np.dot(Iy_win, Iy_win)
            b[0][0] += np.dot(Ix_win, It_win)
            b[1][0] += np.dot(Iy_win, It_win)

            Inverse = np.linalg.pinv(A)
            u[j, i], v[j, i] = np.dot(Inverse, b)

    flow = [u, v]

    return flow
