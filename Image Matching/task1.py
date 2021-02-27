"""
Image Stitching Problem
(Due date: Nov. 9, 11:59 P.M., 2020)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random
import time

def get_smallest(dist, distance, idx):
    if distance[0] < distance[1]:
        if dist < distance[1]:
            distance[1] = dist
            distance[3] = idx

    elif distance[0] > distance[1]:
        if dist < distance[0]:
            distance[0] = dist
            distance[2] = idx

    if distance[0] > distance[1]:
        temp = distance[0]
        distance[0] = distance[1]
        distance[1] = temp
        temp = distance[2]
        distance[2] = distance[3]
        distance[3] = temp

    return distance

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """

    # print(left_img.shape)
    # print(right_img.shape)

    sift = cv2.SIFT_create()
    keypoint_1, descriptor_1 = sift.detectAndCompute(left_img,None)
    keypoint_2, descriptor_2 = sift.detectAndCompute(right_img,None)

    threshold = 0.7
    matches = []
    distance = {}

    for i in range(len(descriptor_1)):
        distance[i] = [100000000,10000000,0,0]
        counter = 0
        for j in range(len(descriptor_2)):
            dist = np.sum(np.subtract(descriptor_1[i],descriptor_2[j])**2)
            distance[i] = get_smallest(dist,distance[i],j)
            if distance[i][0] < threshold*distance[i][1] and counter > 2:
                matches.append([i,distance[i][2]])
                break
            counter += 1

    # matches = []
    # for k in distance.keys():
    #     if distance[k][0] < threshold * distance[k][1]:
    #         matches.append([k,distance[k][2]])

    # print(keypoint_1[1].pt)
    #
    # print(matches)
    # print(distance)
    # print(distance)

    print('done')

    stiched_img = None
    if len(matches) >= 4:
        pts_1 = np.float32([keypoint_1[m[0]].pt for m in matches]).reshape(-1, 1, 2)
        pts_2 = np.float32([keypoint_2[m[1]].pt for m in matches]).reshape(-1, 1, 2)

        M, status = cv2.findHomography(pts_2, pts_1, cv2.RANSAC, 4)
        stiched_img = cv2.warpPerspective(right_img, M, (left_img.shape[1]+right_img.shape[1], left_img.shape[0]))
        stiched_img[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
        # print(retVal.shape)

    return stiched_img

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    start = time.time()
    result_image = solution(left_img, right_img)
    print(time.time() - start)
    cv2.imwrite('results/task1_result.jpg',result_image)


