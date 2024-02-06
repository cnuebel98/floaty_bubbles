import cv2
import numpy as np
import matplotlib.pyplot as plt
from object_tracker import *

# Constants
VIDEO_PATH = r'C:\Users\Carlo\Repos\floaty_bubbles\vids\0_0_60_0_0-10-01-24.avi'
OUTPUT_VIDEO_NAME = 'valves_1_3_5.avi'
WIN_SIZE = (15, 15)

# Dil 5 and ero 15 has the cleanest size graph
MORPH_KERNEL_SIZE_DILATION = 5 
MORPH_KERNEL_SIZE_EROSION = 19 

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
roi = (0, 0, width, height - ignore_region_height)

# Define the codec and create VideoWriter object so that we have the 
# resulting video saved
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, 
                               (width, height - ignore_region_height))

# Smaller epsilon for smoother contours
epsilon_factor = 0.005

# Minimum and maximum diameter for ellipses (150, width/2)
min_ellipse_diameter = 100
max_ellipse_diameter = width/1.5

# Scale bar length in pixels (adjust as needed)
scale_bar_length = 100

# Lists to store data for the diagram
average_sizes = []
num_ellipses = []

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
    total_size = 0
    
    curves_for_frame = [] 
    
    # Draw the specified number of contours with ellipses
    ellipses_for_frame = []  # List to store ellipses for counting
    for i in range(min(num_contours_to_draw, len(contours))):
        contour = contours[i]
        
        #-------------------------------------------------------------
        # determine smoothness of contours
        epsilon = 0.001 * cv2.arcLength(contour, True)
        smooth_contour = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the contour
        cv2.drawContours(frame, [smooth_contour], -1, (255, 0, 0), 1)
        #-------------------------------------------------------------
        
        #-------------------------------------------------------------
        # Fit a polynomial curve to the contour points
        #curve_fit = np.polyfit(contour[:, 0, 0], contour[:, 0, 1], deg=6)
        #curve_points_x = np.linspace(min(contour[:, 0, 0]), max(contour[:, 0, 0]), num=100)
        #curve_points_y = np.polyval(curve_fit, curve_points_x)
        #curve_points = np.column_stack((curve_points_x, curve_points_y))

        # Ensure integer type for drawing
        #curve_points = curve_points.astype(np.int32)

        #curve_points = curve_points.reshape((-1, 1, 2))
        
        # Draw the curve
        #cv2.polylines(frame, [curve_points], isClosed=True, color=(255, 0, 0), thickness=1)
        #-------------------------------------------------------------



        # Fit an ellipse to the contour
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)

            # Calculate major and minor diameters
            major_diameter = max(ellipse[1])
            minor_diameter = min(ellipse[1])

            # Check if the smallest diameter is at least 1/4 of the 
            # largest diameter
            if minor_diameter >= major_diameter / 4:
                # Check if the major axis is at least 100 pixels and 
                # smaller than half the image
                if major_diameter < max_ellipse_diameter:
                    if major_diameter >= min_ellipse_diameter:
                    
                        # Draw the ellipse
                        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

                        # Draw the center of the ellipse as a visible dot
                        center = (int(ellipse[0][0]), int(ellipse[0][1]))
                        detections.append(center)
                        cv2.circle(frame, center, 2, (0, 0, 255), -1)

                        ellipses_for_frame.append(ellipse)

                        # Display the size of the ellipse as text
                        size_text = f"Size: {int(cv2.contourArea(contour))} px^2"
                        cv2.putText(frame, size_text, 
                                    (int(ellipse[0][0]), int(ellipse[0][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (255, 255, 255), 2)

                        # Calculate total size for averaging
                        total_size += cv2.contourArea(contour)

    # 2. Object Tracking
    objects_ids = tracker.update(detections)
    for object_id in objects_ids:
        cx, cy, id = object_id
        cv2.putText(frame, "ID: " + str(id), (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate average size
    if len(ellipses_for_frame) > 0:
        average_size = total_size / len(ellipses_for_frame)
        average_sizes.append(average_size)
        num_ellipses.append(len(ellipses_for_frame))

        # Display live numbers on the frame
        cv2.putText(frame, f"Average Size: {average_size:.2f} px^2", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Number of Ellipses: {len(ellipses_for_frame)}",
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
plt.plot(average_sizes, label='Average Size')
plt.title('Average Size of Ellipses Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Size (px^2)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(num_ellipses, label='Number of Ellipses')
plt.title('Number of Ellipses Over Time')
plt.xlabel('Frame Number')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()
plt.show()