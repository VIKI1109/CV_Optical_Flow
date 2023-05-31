import numpy as np
from LK_pymaid.lk import lucas_kanade_tradition
import cv2

ITERATION = 3
LEVEL = 3
SIGMA = 1.5
KERNEL = (3, 3)


def downSample(matrix):
    gaussian_matrix = cv2.GaussianBlur(matrix, KERNEL, SIGMA, SIGMA)
    downsampled = gaussian_matrix[::2, ::2]

    return downsampled


def lucas_kanade_iterative(firstImage, secondImage, features, u, v, N):

    image_shape = firstImage.shape

    u_previous = np.round(u)
    v_previous = np.round(v)
    window = int(N // 2)

    u = np.zeros(image_shape)
    v = np.zeros(image_shape)

    feature_change_u_array = [0 for i in range(len(features))]
    feature_change_v_array = [0 for i in range(len(features))]

    index = 0

    for l in features:
        j, i = l.ravel()
        j = int(j)
        i = int(i)

        if i in range(window, image_shape[0] - window) and j in range(window, image_shape[
                                                                                  1] - window):

            firstImageFrame = firstImage[
                              i - window: i + window + 1,
                              j - window: j + window + 1]

            # Find indices to warp second image
            up = (i - window) + v_previous[index]
            down = (i + window) + v_previous[index]
            left = (j - window) + u_previous[index]
            right = (j + window) + u_previous[index]

            # Find edge locations and choose possible window
            if (i - window) + v_previous[index] < 0:
                up = 0
                down = N - 1

            if (j - window) + u_previous[index] < 0:
                left = 0
                right = N - 1

            if down > (len(firstImage[:, 0]) - 1):
                up = len(firstImage[:, 0]) - N
                down = len(firstImage[:, 0]) - 1

            if right > (len(firstImage[0, :]) - 1):
                left = len(firstImage[0, :]) - N
                right = len(firstImage[0, :]) - 1

            if np.isnan(up):
                up = i - window
                down = i + window

            if np.isnan(left):
                left = j - window
                right = j + window

            # find the second frame
            secondImageFrame = secondImage[int(up): int(down + 1), int(left): int(right + 1)]

            # Refine optical flow
            u_res, v_res = lucas_kanade_tradition(firstImageFrame, secondImageFrame, N)
            u[i, j] = u_res[window, window] + u_previous[index]
            v[i, j] = v_res[window, window] + v_previous[index]
            feature_change_u_array[index] = u[i, j]
            feature_change_v_array[index] = v[i, j]
            index = index + 1

        else:
            feature_change_u_array[index] = 0
            feature_change_v_array[index] = 0
            index = index + 1

    return feature_change_u_array, feature_change_v_array


def lucas_kanade_pyramid(firstImage, secondImage, features, N, iteration, level_number):

    firstImage = np.array(firstImage)
    secondImage = np.array(secondImage)

    # Create pyramids by downSampling
    firstImagePyramid = np.empty((firstImage.shape[0], firstImage.shape[1], level_number))
    secondImagePyramid = np.empty((secondImage.shape[0], secondImage.shape[1], level_number))
    firstImagePyramid[:, :, 0] = firstImage
    secondImagePyramid[:, :, 0] = secondImage

    for level in range(1, level_number):
        firstImage = downSample(firstImage)
        secondImage = downSample(secondImage)
        firstImagePyramid[0: firstImage.shape[0], 0: firstImage.shape[1], level] = firstImage
        secondImagePyramid[0: secondImage.shape[0], 0: secondImage.shape[1], level] = secondImage

    level0 = level_number - 1

    feature_point_level0 = features // 2 ** level0

    firstImage_level0 = firstImagePyramid[
                        0: (len(firstImagePyramid[:, 0]) // 2 ** level0),
                        0: (len(firstImagePyramid[0, :]) // 2 ** level0), level0
                        ]
    secondImage_level0 = secondImagePyramid[
                         0: (len(secondImagePyramid[:, 0]) // 2 ** level0),
                         0: (len(secondImagePyramid[0, :]) // 2 ** level0),
                         level0,
                         ]

    u = [0 for i in range(len(features))]
    v = [0 for i in range(len(features))]

    for i in range(0, iteration):
        (u, v) = lucas_kanade_iterative(firstImage_level0, secondImage_level0, feature_point_level0, u, v, N)


    # Find optical flow of all levels of pyramid

    for i in range(1, level_number):
        unsampled_u = [i * 2 for i in u]
        unsampled_v = [j * 2 for j in v]
        levels = level_number - i - 1
        feature_point = np.array([i // 2 ** levels for i in features])
        firstImageInLevelPyramid = firstImagePyramid[
                                 0: (len(firstImagePyramid[:, 0]) // 2 ** levels),
                                 0: (len(firstImagePyramid[0, :]) // 2 ** levels),
                                 levels,
                                 ]
        secondImageInLevelPyramid = secondImagePyramid[
                                  0: (len(secondImagePyramid[:, 0]) // 2 ** levels),
                                  0: (len(secondImagePyramid[0, :]) // 2 ** levels),
                                  levels,
                                  ]

        (u, v) = lucas_kanade_iterative(firstImageInLevelPyramid, secondImageInLevelPyramid, feature_point, unsampled_u,
                                        unsampled_v, N)


    return u, v
