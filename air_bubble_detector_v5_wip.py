import cv2
import numpy as np
import matplotlib.pyplot as plt
from object_tracker import *

# Constants
VIDEO_PATH = r'C:\Users\Carlo\Repos\floaty_bubbles\vids\0_60_0_60_0-10-1-24.avi'
OUTPUT_VIDEO_NAME = 'valves_1_3_5.avi'
WIN_SIZE = (15, 15)

# Dil 5 and ero 15 has the cleanest size graph
MORPH_KERNEL_SIZE_DILATION = 5 
MORPH_KERNEL_SIZE_EROSION = 15 

# Create tracker object
tracker = EuclideanDistTracker()

#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Get the video
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ignore bottom 20 mm because of the bottom video properties when 
# playing the video
ignore_region_height = int(20 / 25.4 * cap.get(cv2.CAP_PROP_FPS))
roi = (20, 20, width, height - ignore_region_height)

# Define the codec and create VideoWriter object so that we have the 
# resulting video saved
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, 
                               (width, height - ignore_region_height))

# Scale bar length in pixels (adjust as needed)
scale_bar_length = 100

# Lists to store data for the diagram
average_sizes_contours = []
num_contours_per_frame = []

# Read the first frame from the video
ret, _ = cap.read()

# Counter for frame number
frame_number = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply ROI
    frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

    #frame = cv2.GaussianBlur(frame, (11, 11), 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normal Thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological Operations (Dilation followed by Erosion)
    morph_kernel_dil = np.ones((MORPH_KERNEL_SIZE_DILATION, MORPH_KERNEL_SIZE_DILATION), np.uint8)
    morph_kernel_ero = np.ones((MORPH_KERNEL_SIZE_EROSION, MORPH_KERNEL_SIZE_EROSION), np.uint8)
    morphed_thresh = cv2.dilate(thresh, morph_kernel_dil, iterations=1)
    morphed_thresh = cv2.erode(morphed_thresh, morph_kernel_ero, iterations=1)

    #morphed_thresh = cv2.GaussianBlur(morphed_thresh, (21, 21), 0)

    # Find contours in the inverse of the morphed thresholded image
    contours, _ = cv2.findContours(~morphed_thresh, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    # Sort contours by area or intensity
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    num_contours_to_draw = min(10, len(contours))
    total_size_contour = 0
    curves_for_frame = [] 
    
    contours_for_counting = []
    
    #for i in range(min(num_contours_to_draw, len(contours))):
    for i in range(0, num_contours_to_draw):
        contour = contours[i]
        
        if (len(contour) >= 5 and cv2.contourArea(contour) > 2000  
            and cv2.contourArea(contour)< 300000):

                #-------------------------------------------------------------
                # determine smoothness of contours
                epsilon = 0.0001 * cv2.arcLength(contour, True)
                smooth_contour = cv2.approxPolyDP(contour, epsilon, True)

                # Calculate centroids of Contours
                M = cv2.moments(contour)
                
                # Calculate centroid using moments
                if M["m00"] != 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                else:
                    centroid_x, centroid_y = 0, 0

                cv2.drawContours(frame, [smooth_contour], -1, (255, 0, 0), 2)
                
                center_contour = (int(centroid_x), int(centroid_y))

                detections.append(center_contour)

                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

                #ellipses_for_frame.append(ellipse)
                contours_for_counting.append(contour)
                
                # Display the size of the contour as text
                size_text = f"Size Contour: {int(cv2.contourArea(contour))} px^2"
                cv2.putText(frame, size_text, 
                            (int(centroid_x), int(centroid_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (255, 255, 255), 2)
                
                # Calculate total size for averaging
                total_size_contour += cv2.contourArea(contour)

    # 2. Object Tracking
    objects_ids = tracker.update(detections)
    for object_id in objects_ids:
        cx, cy, id = object_id
        cv2.putText(frame, "ID: " + str(id), (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate average size
    if len(contour) > 0:
        
        average_size_contour = total_size_contour / len(contour)
        average_sizes_contours.append(average_size_contour)
        num_contours_per_frame.append(len(contours_for_counting))
        
        # Display live numbers on the frame
        cv2.putText(frame, f"Average Size Contours/Frame: {average_size_contour:.2f} px^2", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Number of Contours: {len(contours_for_counting)}",
                     (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                     0.8, (255, 255, 255), 2)

    # Add a scale bar at the top-right corner
    scale_bar_start = (width - 10 - scale_bar_length, 10)
    scale_bar_end = (width - 10, 10)
    cv2.line(frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)

    # Save Processed Frame to Output Video
    output_video.write(frame)

    # Visualization
    cv2.imshow('preprocessing', ~morphed_thresh)
    cv2.imshow('Processed Frame', frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

# Release Resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Plotting the diagram
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(average_sizes_contours, label='Average Size Contours')
plt.title('Average Size of Contours Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Size (px^2)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(num_contours_per_frame, label='Number of Contours')
plt.title('Number of Contours per frame')
plt.xlabel('Frame Number')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()
plt.show()