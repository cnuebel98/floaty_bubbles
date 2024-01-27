import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Constants
VIDEO_PATH = r'C:\Users\Carlo\Repos\floaty_bubbles\vids\0_60_0_60_0-10-1-24.avi'
OUTPUT_VIDEO_NAME = 'valves_1_3_5.avi'
WIN_SIZE = (15, 15)
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_CONSTANT = 2
MORPH_KERNEL_SIZE_DILATION = 5
MORPH_KERNEL_SIZE_EROSION = 15

# Global variable for index counter
global_index_counter = 0

movement_distance_threshold = 20

# Get the video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Unable to open video file.")

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

# Minimum and maximum diameter for ellipses (150, width/2)
min_ellipse_diameter = 100
max_ellipse_diameter = width/1.5

# Scale bar length in pixels (adjust as needed)
scale_bar_length = 100

# Lists to store data for the diagram
average_sizes = []
num_ellipses = []

# Process Frames with Adaptive Thresholding, Morphological Operations, 
# and Contour Detection
# Read the first frame from the video
ret, prev_frame = cap.read()

# Counter for frame number
frame_number = 0  



# List to store ellipses for tracking
prev_ellipses = []

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
        
        # Draw the contour
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 1)
        
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

                        # Store the ellipse as a dictionary with 'center' key
                        ellipse_data = {'center': center, 'index': i}
                        ellipses_for_frame.append(ellipse_data)

                        # Display the size of the ellipse as text
                        size_text = f"Size: {int(cv2.contourArea(contour))} px^2"
                        cv2.putText(frame, size_text, 
                                    (int(ellipse[0][0]), int(ellipse[0][1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (255, 255, 255), 2)
                        
                        # Display the index of the ellipse underneath the size
                        index_text = f"Index: {ellipse_data['index']}"  # Use ellipse_data instead of curr_ellipse
                        cv2.putText(frame, index_text, 
                                    (int(ellipse[0][0]), int(ellipse[0][1]) + 20),  # Adjust the vertical position as needed
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (255, 255, 255), 2)

                        # Calculate total size for averaging
                        total_size += cv2.contourArea(contour)
    
   # Track ellipses using the Hungarian algorithm
    if len(prev_ellipses) > 0 and len(ellipses_for_frame) > 0:
        cost_matrix = np.zeros((len(prev_ellipses), len(ellipses_for_frame)))
        for i, prev_ellipse in enumerate(prev_ellipses):
            for j, curr_ellipse in enumerate(ellipses_for_frame):
                cost_matrix[i, j] = np.linalg.norm(np.array(prev_ellipse['center']) - np.array(curr_ellipse['center']))

        prev_indices, curr_indices = linear_sum_assignment(cost_matrix)

        # Dictionary to store mapping between ellipse indices and unique identifiers
        ellipse_index_mapping = {}

        # Update ellipses with matching indices
        matched_ellipses = []

        for prev_index, curr_index in zip(prev_indices, curr_indices):
            prev_ellipse = prev_ellipses[prev_index]
            curr_ellipse = ellipses_for_frame[curr_index]

            # For example, you might want to store the movement distance
            movement_distance = np.linalg.norm(np.array(prev_ellipse['center']) - np.array(curr_ellipse['center']))

            # Check if the movement distance is below the threshold
            if movement_distance < movement_distance_threshold:
                # If the current ellipse is new, assign a new unique identifier
                if curr_index not in ellipse_index_mapping:
                    ellipse_index_mapping[curr_index] = global_index_counter
                    global_index_counter += 1

                # Store additional information in the current ellipse data
                curr_ellipse['movement_distance'] = movement_distance
                curr_ellipse['index'] = ellipse_index_mapping[curr_index]

                # Store matched ellipse for future tracking
                matched_ellipses.append(curr_ellipse)
            else:
                # If movement distance is above threshold, consider it as a new ellipse
                curr_ellipse['movement_distance'] = movement_distance
                curr_ellipse['index'] = global_index_counter
                global_index_counter += 1

                # Store the new ellipse for future tracking
                matched_ellipses.append(curr_ellipse)

        # Identify unmatched ellipses and mark them with a special value
        unmatched_indices = set(range(len(ellipses_for_frame))) - set(curr_indices)
        for unmatched_index in unmatched_indices:
            unmatched_ellipse = ellipses_for_frame[unmatched_index]
            unmatched_ellipse['movement_distance'] = 0  # Set a value indicating no movement
            unmatched_ellipse['index'] = global_index_counter  # Assign a new unique identifier
            global_index_counter += 1  # Increment the global counter

            # Store the unmatched ellipse for future tracking
            matched_ellipses.append(unmatched_ellipse)

        # Update previous ellipses for the next iteration
        prev_ellipses = matched_ellipses
            
    else:
        # If it's the first frame, just use the current ellipses for tracking
        prev_ellipses = ellipses_for_frame
        
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