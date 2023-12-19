import imageio.v2 as img2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

if __name__ == '__main__':
    # read image
    flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(r'C:\Users\Carlo\Repos\floaty_bubbles\pics\IMG_5987.png', flag)
    print("version: " + str(cv2.__version__))
    # resize image
    size = [500, 500]
    image = cv2.resize(img, size)
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(image)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),  
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #show image
    cv2.imshow("Keypoints", im_with_keypoints)
    #cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()