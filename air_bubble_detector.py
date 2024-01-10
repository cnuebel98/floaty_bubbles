import cv2
import numpy as np

# Updated video path
video_path = r'C:\Users\Carlo\Repos\floaty_bubbles\vids\0_0_60_0_0-10-01-24.avi'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region to ignore (bottom 20 mm)
ignore_region_height = int(20 / 25.4 * cap.get(cv2.CAP_PROP_FPS))  # Convert 20 mm to pixels at 25.4 mm/inch
roi = (0, 0, width, height - ignore_region_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_contours_smoothed_morphology_custom_params.avi', fourcc, fps, (width, height - ignore_region_height))

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Smoothness of contours (higher is more edgy)
epsilon_factor = 0.0002

# Number of contours to draw
num_contours_to_draw = 3

# Morphological operations kernel size
morph_kernel_size = 2

# Step 3: Process Frames with Adaptive Thresholding, Morphological Operations, and Contour Detection
ret, prev_frame = cap.read()
prev_frame = prev_frame[roi[1]:roi[3], roi[0]:roi[2]]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply ROI
    frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Morphological Operations (Dilation followed by Erosion)
    morph_kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    morphed_thresh = cv2.dilate(adaptive_thresh, morph_kernel, iterations=1)
    morphed_thresh = cv2.erode(morphed_thresh, morph_kernel, iterations=1)

    # Find contours in the inverse of the morphed thresholded image
    contours, _ = cv2.findContours(~morphed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area or intensity (you can choose either)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Draw the specified number of contours with the highest contrast (smoothed)
    for i in range(min(num_contours_to_draw, len(contours))):
        epsilon = epsilon_factor * cv2.arcLength(contours[i], True)
        smoothed_contour = cv2.approxPolyDP(contours[i], epsilon, True)
        cv2.drawContours(frame, [smoothed_contour], -1, (0, 255, 0), 2)

    # Step 6: Save Processed Frame to Output Video
    output_video.write(frame)

    # Step 6: Visualization (Optional)
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 7: Release Resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
