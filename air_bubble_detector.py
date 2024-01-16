import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
VIDEO_PATH = r'C:\Users\Carlo\Repos\floaty_bubbles\vids\0_0_60_0_0-10-01-24.avi'
OUTPUT_VIDEO_NAME = 'valves_1_3_5.avi'
WIN_SIZE = (15, 15)
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_CONSTANT = 2
MORPH_KERNEL_SIZE = 2

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

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=WIN_SIZE, maxLevel=2, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Smaller epsilon for smoother contours
epsilon_factor = 0.005

# Minimum and maximum diameter for ellipses
min_ellipse_diameter = 150
max_ellipse_diameter = width / 2

# Scale bar length in pixels (adjust as needed)
scale_bar_length = 150

# Lists to store data for the diagram
average_sizes = []
num_ellipses = []

# Process Frames with Adaptive Thresholding, Morphological Operations, 
# and Contour Detection
# Read the first frame from the video
ret, prev_frame = cap.read()

# Apply the region of interest (ROI) to the first frame
prev_frame = prev_frame[roi[1]:roi[3], roi[0]:roi[2]]

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize ellipses for tracking
ellipses_for_tracking = []

# Counter for frame number
frame_number = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply ROI
    frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_CONSTANT
    )

    # Morphological Operations (Dilation followed by Erosion)
    morph_kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    morphed_thresh = cv2.dilate(adaptive_thresh, morph_kernel, iterations=1)
    morphed_thresh = cv2.erode(morphed_thresh, morph_kernel, iterations=1)

    # Find contours in the inverse of the morphed thresholded image
    contours, _ = cv2.findContours(~morphed_thresh, 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area or intensity
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    num_contours_to_draw = min(10, len(contours))
    total_size = 0

    # Draw the specified number of contours with ellipses
    ellipses_for_frame = []  # List to store ellipses for counting
    for i in range(min(num_contours_to_draw, len(contours))):
        contour = contours[i]

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
                        cv2.circle(frame, center, 2, (0, 0, 255), -1)

                        # Save the ellipse for tracking
                        ellipses_for_tracking.append(ellipse)
                        ellipses_for_frame.append(ellipse)

                        # Display the size of the ellipse as text
                        size_text = f"Size: {
                            int(cv2.contourArea(contour))} px^2"
                        cv2.putText(frame, size_text, 
                                    (int(ellipse[0][0]), int(ellipse[0][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (255, 255, 255), 2)

                        # Calculate total size for averaging
                        total_size += cv2.contourArea(contour)

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
    cv2.imshow('Processed Frame', frame)
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
