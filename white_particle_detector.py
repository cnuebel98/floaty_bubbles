import cv2
import numpy as np

# Step 2: Load Video
video_path = r'C:\Users\Carlo\Repos\floaty_bubbles\vids\0_0_60_0_0-10-01-24.avi'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the region to ignore (bottom 5 mm)
ignore_region_height = int(20 / 25.4 * cap.get(cv2.CAP_PROP_FPS))  # Convert 5 mm to pixels at 25.4 mm/inch
roi = (0, 0, width, height - ignore_region_height)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video_with_contours.avi', fourcc, fps, (width, height - ignore_region_height))

# Parameters for Lucas-Kanade Optical Flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Step 3: Process Frames with Adaptive Thresholding
ret, prev_frame = cap.read()
prev_frame = prev_frame[roi[1]:roi[3], roi[0]:roi[2]]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Define some random points as the starting points for Optical Flow
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=10000, qualityLevel=0.2, minDistance=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply ROI
    frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Calculate optical flow
    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, adaptive_thresh, p0, None, **lk_params)

    # Select good points
    good_old = p0[status == 1]
    good_new = p1[status == 1]

    # Draw the tracks
    for i in range(len(good_new)):
        a, b = good_new[i].ravel()
        c, d = good_old[i].ravel()
        cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

    # Step 6: Save Processed Frame to Output Video
    output_video.write(frame)

    # Step 6: Visualization (Optional)
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update previous frame and points
    prev_gray = adaptive_thresh.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Step 7: Release Resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
