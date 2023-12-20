import imageio.v2 as img2
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

def adaptive_thresholding(image):
    '''returns the adaptive_threshold image'''
    adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 7)
    return adaptive_threshold


if __name__ == '__main__':
    # read image
    flag = cv2.IMREAD_GRAYSCALE
    img = cv2.imread(r'C:\Users\Carlo\Repos\floaty_bubbles\pics\IMG_5987.png', flag)
    print("version: " + str(cv2.__version__))
    # resize image
    size = [500, 500]
    image = cv2.resize(img, size)
    params = cv2.SimpleBlobDetector_Params()

    # Parameter manipulation
    
    adaptive_threshold = adaptive_thresholding(image)

    #threshold_values = [1, 10, 50, 100, 150, 500, 1000]
    threshold_values = [10]

    # Range of parameter values to iterate over
    min_circularity_values = [0.1]
    min_convexity_values = [0.2]
    min_inertia_ratio_values = [0.1]
    min_area_values = [20]

    for min_area in min_area_values:
        for min_circularity in min_circularity_values:
            for min_convexity in min_convexity_values:
                for min_inertia_ratio in min_inertia_ratio_values:
                    for threshold in threshold_values:
                        params.minThreshold = threshold
                        params.maxThreshold = 255

                        # Filter for black blobs
                        params.filterByColor = False
                        params.blobColor = 255  # 0 for black, 255 for white

                        # Filter by area
                        params.filterByArea = True
                        params.minArea = min_area  # Adjust as needed
                        params.maxArea = 10000  # Adjust as needed

                        # Filter by circularity
                        params.filterByCircularity = True
                        params.minCircularity = min_circularity  # Adjust as needed

                        # Filter by convexity
                        params.filterByConvexity = True
                        params.minConvexity = min_convexity  # Adjust as needed

                        # Filter by inertia
                        params.filterByInertia = True
                        params.minInertiaRatio = min_inertia_ratio  # Adjust as needed

                        detector = cv2.SimpleBlobDetector_create(params)
                        keypoints = detector.detect(image)


                        # Filter out keypoints with insufficient information
                        #keypoints = [keypoint for keypoint in keypoints if keypoint.size >= 1]
                        
                        # Convert keypoints to NumPy array for clustering
                        points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)

                        # Use the Elbow method to find the optimal number of clusters
                        #distortions = []
                        #K_range = range(1, min(len(points), 10))  # Limit the range for a reasonable number of clusters
                        #for k in K_range:
                        #    _, _, dist = cv2.kmeans(points, k, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
                        #    distortions.append(dist)

                        # Plot the Elbow curve
                        #plt.plot(K_range, distortions, marker='o')
                        #plt.xlabel('Number of clusters')
                        #plt.ylabel('Distortion')
                        #plt.title('Elbow Method for Optimal k')
                        #plt.show()

                        # Determine the optimal number of clusters (e.g., manually or programmatically)
                        #optimal_k = int(input("Enter the optimal number of clusters: "))

                        # Define criteria and apply kmeans()
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                        _, labels, centers = cv2.kmeans(points, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                        # Draw ellipses around each cluster
                        result_0 = image.copy()
                        #_, labels, centers = cv2.kmeans(points, optimal_k, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
                        for i in range(len(centers)):
                            cluster_points = points[labels.flatten() == i]
                            if len(cluster_points) >= 3:
                                # Create convex hull from cluster points
                                hull = cv2.convexHull(cluster_points, clockwise=False)
                                # Fit ellipse to convex hull
                                ellipse = cv2.fitEllipse(hull)
                                # Draw ellipse on the result image

                                # Fit ellipse using fitEllipseDirect
                                #rect = cv2.fitEllipseDirect(cluster_points)
                                # Convert rect to ellipse format (center, axes, angle)
                                #ellipse = ((rect[0][0], rect[0][1]), (rect[1][0] / 2, rect[1][1] / 2), rect[2])
                                #ellipse = cv2.fitEllipse(cluster_points)
                                cv2.ellipse(result_0, ellipse, (0, 0, 255), 2)

                        # Postprocessing step for merging close blobs with each other
                        #merge_threshold = 30
                        #merged_keypoints = []

                        #for i, keypoint_i in enumerate(keypoints):
                        #    for j, keypoint_j in enumerate(keypoints):
                        #        if i != j and np.linalg.norm(np.array(keypoint_i.pt) - np.array(keypoint_j.pt)) < merge_threshold:
                        #            # Merge the two keypoints
                        #            merged_keypoint = cv2.KeyPoint(
                        #                (keypoint_i.pt[0] + keypoint_j.pt[0]) / 2,
                        #                (keypoint_i.pt[1] + keypoint_j.pt[1]) / 2,
                        #                keypoint_i.size + keypoint_j.size,
                        #            )
                        #            merged_keypoints.append(merged_keypoint)

                        # Add the keypoints that are not merged
                        #merged_keypoints.extend(keypoints)

                        im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),  
                                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        #im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),  
                        #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        # Fit ellipses to keypoints
                        #print(len(merged_keypoints))
                        print(len(keypoints))
                        try:
                            if len(keypoints) >= 5:
                                ellipses = []
                                for keypoint in keypoints:
                                    ...
                                    #ellipse = cv2.fitEllipse(np.array(keypoint.pt).reshape(-1, 1, 2).astype(np.int32))
                                    #ellipses.append(ellipse)

                                # Draw ellipses on the image
                                result = im_with_keypoints.copy()
                                for ellipse in ellipses:
                                    ...
                                    #cv2.ellipse(result, ellipse, (0, 0, 255), 2)

                                #show image
                                print("---------------------------")
                                print("area: " + str(min_area))
                                print("circularity: " + str(min_circularity))
                                print("convexity: " + str(min_convexity))
                                print("intertia: " + str(min_inertia_ratio))
                                print("threshold: " + str(threshold))
                                cv2.imshow("Keypoints", im_with_keypoints)
                                #cv2.imshow('image', result)
                                cv2.waitKey(50)
                        except cv2.error as e:
                            print(f"Error: {e}")
    cv2.imshow("clustered", result_0)
    #cv2.imshow("Original Image", image)
    #cv2.imshow("Adaptive Thresholding", adaptive_threshold)
    cv2.waitKey()
    cv2.destroyAllWindows()