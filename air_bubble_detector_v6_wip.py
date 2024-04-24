import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from object_tracker import *

folder_path = r'D:\testing'
# Create an empty DataFrame
results_df = pd.DataFrame(columns=['video_name', 'frame', 'num_contours', 'center_of_mass', 'id_x_pos', 'id_y_pos', 'size_per_bubble', 'perimeter_per_bubble'])


for file_name in os.listdir(folder_path):
    if file_name.endswith('.avi'):  # Adjust the file extensions as needed
        VIDEO_PATH = os.path.join(folder_path, file_name)
        
        print(file_name)
        # Process the video here
        
        # Constants
        #VIDEO_PATH = r'D:\V30_20240418_125938_20240418_130046.avi'
        OUTPUT_VIDEO_NAME = 'umf_1_0.avi'
        WIN_SIZE = (15, 5)

        # Get the video
        cap = cv2.VideoCapture(VIDEO_PATH)

        # Check if the video file opened successfully
        if not cap.isOpened():
            print(f"Error opening {file_name}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create tracker object
        tracker = EuclideanDistTracker(width)

        # Ignore bottom 20 mm because of the bottom video properties when 
        # playing the video
        ignore_region_height = int(47 / 25.4 * cap.get(cv2.CAP_PROP_FPS))
        roi = (88, 0, width-55, height - ignore_region_height)

        # Define the codec and create VideoWriter object so that we have the 
        # resulting video saved
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, 
                                    (width, height - ignore_region_height))

        # Scale bar length in pixels (adjust as needed)
        scale_bar_length = 8

        # Lists to store data for the diagram
        average_sizes_contours = []
        total_sizes_contours = []
        num_contours_per_frame = []
        ids_and_sizes = {}
        first_detected_frame = {}
        
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

            # Greyscaling of the image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Normal Thresholding
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Create a circular kernels
            radius_dil = 2
            radius_ero = 6

            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius_dil+1, 2*radius_dil+1))
            kernel_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius_ero+1, 2*radius_ero+1))
            
            # Ero and dil with circular kernel
            morphed_thresh_dil = cv2.dilate(thresh, kernel_dil, iterations = 1)
            morphed_thresh_ero = cv2.erode(morphed_thresh_dil, kernel_ero, iterations = 1)

            # Find contours in the inverse of the morphed thresholded image
            contours, _ = cv2.findContours(~morphed_thresh_ero, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            # Sort contours by area or intensity
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            num_contours_to_draw = min(10, len(contours))
            total_size_contour = 0
            #curves_for_frame = []
            contours_for_counting = []
            size_list = []
            perimeter_list = []
            #for i in range(min(num_contours_to_draw, len(contours))):
            for i in range(0, num_contours_to_draw):
                contour = contours[i]
                
                if (len(contour) >= 5 and cv2.contourArea(contour) > 500  
                    and cv2.contourArea(contour) < 150000):

                        #-------------------------------------------------------------
                        # determine smoothness of contours
                        epsilon = 0.0005 * cv2.arcLength(contour, True)
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
                        size_list.append(round(cv2.contourArea(contour), 4))
                        perimeter_list.append(round(cv2.arcLength(contour, True), 4))
                        total_size_contour += cv2.contourArea(contour)

            

            # 2. Object Tracking
            objects_ids = tracker.update(detections, frame_number)
            dict_cx = {}
            dict_cy = {}
            dict_center_of_mass = {}
            
            for object_id in objects_ids:
                cx, cy, id = object_id
                dict_cx[id] = cx
                dict_cy[id] = cy
                dict_center_of_mass[id] = (cx, cy)
                cv2.putText(frame, "ID: " + str(id), (cx, cy+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


            # Calculate average size
            if len(contour) > 0:
                
                average_size_contour = total_size_contour / len(contour)
                average_sizes_contours.append(average_size_contour)
                total_sizes_contours.append(total_size_contour)
                num_contours_per_frame.append(len(contours_for_counting))
                
                for object_id, size in zip(objects_ids, average_sizes_contours):
                    _, _, id = object_id
                    if id not in ids_and_sizes:
                        ids_and_sizes[id] = size
                        first_detected_frame[id] = frame_number

                # Display live numbers on the frame
                cv2.putText(frame, f"Average Size Contours/Frame: {average_size_contour:.2f} px^2", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Number of Contours: {len(contours_for_counting)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255, 255, 255), 2)

            # Add a scale bar at the top-right corner
            scale_bar_start = (width-307 - scale_bar_length, 605)
            scale_bar_end = (width-307, 605)

            #scale_bar_start = (width - 10 - scale_bar_length, 10)
            #scale_bar_end = (width - 10, 10)

            #cv2.line(frame, scale_bar_start, scale_bar_end, (255, 255, 255), 2)

            # Save Processed Frame to Output Video
            output_video.write(frame)

            # Visualization
            cv2.imshow('dil+ero', ~morphed_thresh_ero)
            cv2.imshow('dil', ~morphed_thresh_dil)
            
            cv2.imshow('Processed Frame', frame)

            key = cv2.waitKey(30)
            if key == 27:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            #print("---------------")
            
            #print(frame_number)
            #print(dict_center_of_mass)
            #print(perimeter_list)
            #print(size_list)
            
            new_row = {'video_name': file_name,'frame': frame_number, 'num_contours': num_contours_per_frame[-1], 'center_of_mass': dict_center_of_mass, 'id_x_pos': dict_cx, 'id_y_pos': dict_cy, 'size_per_bubble': size_list, 'perimeter_per_bubble': perimeter_list}
            results_df = results_df._append(new_row, ignore_index=True)
            #print(results_df)
            frame_number += 1

        # Release Resources
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

results_df.to_csv("results_df.csv")

# Plotting the diagram
plt.figure(figsize=(12, 6))
plt.subplot(4, 1, 1)
plt.plot(total_sizes_contours, "o", label='Total Size Contours')
plt.title('Total Size of Contours per frame')
plt.xlabel('Frame Number')
plt.ylabel('Size (px^2)')
# Annotate the plot with IDs at the corresponding frame on the x-axis where they are first detected
for id, size in ids_and_sizes.items():
    plt.text(first_detected_frame[id], size, str(id), fontsize=14, color='red')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(num_contours_per_frame, label='Number of Contours')
plt.title('Number of Contours per frame')
plt.xlabel('Frame Number')
plt.ylabel('Count')
plt.legend()

plt.subplot(4, 1, 3)
tracker.plot_distances()

plt.subplot(4, 1, 4)
tracker.plot_total_distances()


plt.figure(figsize=(12,6))
plt.subplot(2, 1, 1)
tracker.plot_x_movement()

plt.subplot(2, 1, 2)
tracker.plot_y_movement()

plt.tight_layout()
#plt.show()