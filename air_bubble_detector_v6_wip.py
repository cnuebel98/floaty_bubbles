import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from object_tracker import *


# All Videos in the folder will be analysed and the extracted data will 
# be put into one dataframe
folder_path = r'D:\testing'

# Create an empty results DataFrame
results_df = pd.DataFrame(columns=['video_name', 
                                   'frame_number', 
                                   'time [sec]', 
                                   'num_contours_per_frame', 
                                   'ID', 
                                   'cx_pos [px]', 
                                   'cy_pos [px]', 
                                   'cx_pos [m]',
                                   'cy_pos [m]', 
                                   'size [px^2]', 
                                   'size [m^2]', 
                                   'perimeter [px]',
                                   'perimeter [m]',
                                   'approx_diameter [px]',
                                   'approx_diameter [m]',
                                   'x_velocity [m/s]',
                                   'y_velocity [m/s]'])

# iterates over all Videos in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.avi'):  # Adjust the file extensions as needed
        VIDEO_PATH = os.path.join(folder_path, file_name)
        
        print(file_name)
        time = 0
        
        # Constants
        #VIDEO_PATH = r'D:\V30_20240418_125938_20240418_130046.avi'
        #OUTPUT_VIDEO_NAME = 'umf_1_0.avi'
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
        #output_video = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, fps, 
        #                            (width, height - ignore_region_height))

        # Scale bar length in pixels (adjust as needed)
        scale_bar_length = 30

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
            
            # 100 cm is 1920 px
            # 10 cm is 192 px
            # 1 cm is 19.2 px
            
            # Add a scale bar at the top-right corner
            scale_bar_start = (20, 430-192)
            scale_bar_end = (20, 430)

            #scale_bar_start = (width - 10 - scale_bar_length, 10)
            #scale_bar_end = (width - 10, 10)
            sc_start = (30, 430-384)
            sc_end = (30, 430)

            cv2.line(frame, scale_bar_start, scale_bar_end, (0, 0, 255), 2)
            cv2.line(frame, sc_start, sc_end, (0, 0, 255), 2)

            #cv2.imshow('testing', frame)
            # Apply ROI
            frame = frame[roi[1]:roi[3], roi[0]:roi[2]]
            time = frame_number*(1/50)

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

            # Save Processed Frame to Output Video
            #output_video.write(frame)

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

            #print(dict_center_of_mass)
            
            list_counter = 0
            for id in dict_center_of_mass:
                
                # X and Y coordinates of center of bubble
                cx = dict_center_of_mass[id][0]
                cy = dict_center_of_mass[id][1]

                # diameter if bubble were a circle
                approx_diameter = 2*math.sqrt(((size_list[list_counter])/math.pi))

                # find ID in previous frame
                # look through the last 20 rows of the dataframe to find
                # the cx and cy position of the current bubble
                search_radius = 20
                x_velocity_meter_per_sec = 0
                y_velocity_meter_per_sec = 0
                # only works starting with the second frame
                if frame_number > 0:
                    prev_frame = frame_number - 1
                    # go through last 20 or through the whole df if it 
                    # has less than 20 entries
                    for x in range(0, min(search_radius, len(results_df))):
                        # if the ID we are looking for is found in one 
                        # of the previous rows and also if the frame 
                        # number in that row is from the previous frame
                        if id == results_df.iloc[-x]['ID'] and frame_number == (results_df.iloc[-x]['frame_number']+1):
                            # velocities are calculated in px first 
                            #(just current minus previous center 
                            #positions) and the redone in meter (/1920)
                            x_velocity_px_per_frame = (cx - results_df.iloc[-x]['cx_pos [px]'])
                            y_velocity_px_per_frame = (cy - results_df.iloc[-x]['cx_pos [px]'])
                            x_velocity_meter_per_frame = x_velocity_px_per_frame/1920
                            y_velocity_meter_per_frame = y_velocity_px_per_frame/1920
                            x_velocity_meter_per_sec = x_velocity_meter_per_frame*50
                            y_velocity_meter_per_sec = y_velocity_meter_per_frame*50

                            #print(str(id) + " had prev x pos: " + str(results_df.iloc[-x]['cx_pos']))
                            #print(str(id) + " had prev y pos: " + str(results_df.iloc[-x]['cy_pos']))
                            #print(str(id) + " now has x pos: " + str(cx))
                            #print(str(id) + " now has y pos: " + str(cy))
                            #print(x_velocity)
                            #print(y_velocity)
                
                # No velocity fro first frame can be calculated
                else: 
                    x_velocity_meter_per_sec, y_velocity_meter_per_sec = 0, 0
                    x_velocity_px_per_frame, y_velocity_px_per_frame = 0, 0

                # all the Information trackable so far
                #print("-----------")
                
                
                #print(size_list[list_counter]/(1920**2))
                #print((size_list[list_counter]*625/2304)*10**(-6))
               # print("frame_number: " + str(frame_number) + 
               #       ", time [sec]: " + str(time) + 
               #       ", video_name: " + str(file_name) + 
               #       ', total_num_contours: ' + str(num_contours_per_frame[-1]) + 
               #       ", ID: " + str(id) + 
               #       ", CX: " + str(cx) + 
               #       ", CY: " + str(cy) + 
               #       ", size [px^2]: " + str(size_list[list_counter]) +
               #       ", size [m^2]: " + str(round(size_list[list_counter]/(1920**2), 8)) + 
               #       ", perimeter [px] : " + str(perimeter_list[list_counter]) + 
               #       ", perimeter [m]: " + str(round(perimeter_list[list_counter]/1920, 4)) +
               #       ", approx. diameter: " + str(round(approx_diameter, 4)))
                
                new_row = {'video_name': file_name,
                           'frame_number': frame_number, 
                           'time [sec]': time, 
                           'num_contours_per_frame': num_contours_per_frame[-1],
                           'ID': id,
                           'cx_pos [px]': cx, 
                           'cy_pos [px]': cy,
                           'cx_pos [m]': round(cx/1920, 4), 
                           'cy_pos [m]': round(cy/1920, 4), 
                           'size [px^2]': size_list[list_counter],
                           'size [m^2]': round(size_list[list_counter]/(1920**2), 8),
                           'perimeter [px]': perimeter_list[list_counter],
                           'perimeter [m]': round(perimeter_list[list_counter]/1920, 4),
                           'approx_diameter [px]': round(approx_diameter, 4),
                           'approx_diameter [m]': round(approx_diameter/1920, 4),
                           'x_velocity [m/s]': round(x_velocity_meter_per_sec, 4),
                           'y_velocity [m/s]': round(y_velocity_meter_per_sec, 4)}
                results_df = results_df._append(new_row, ignore_index=True)

                list_counter += 1

            #new_row = {'video_name': file_name,'frame': frame_number, 'time [sec]': time, 'num_contours': num_contours_per_frame[-1], 'center_of_mass': dict_center_of_mass, 'id_x_pos': dict_cx, 'id_y_pos': dict_cy, 'size_per_bubble': size_list, 'perimeter_per_bubble': perimeter_list}
            #results_df = results_df._append(new_row, ignore_index=True)
            #print(results_df)
            frame_number += 1

        # Release Resources
        cap.release()
        #output_video.release()
        cv2.destroyAllWindows()

    # Export the results dataframe
    results_df.to_csv(str(file_name) + "_results_df.csv")
    results_df = pd.DataFrame(columns=['video_name', 
                                   'frame_number', 
                                   'time [sec]', 
                                   'num_contours_per_frame', 
                                   'ID', 
                                   'cx_pos [px]', 
                                   'cy_pos [px]', 
                                   'cx_pos [m]',
                                   'cy_pos [m]', 
                                   'size [px^2]', 
                                   'size [m^2]', 
                                   'perimeter [px]',
                                   'perimeter [m]',
                                   'approx_diameter [px]',
                                   'approx_diameter [m]',
                                   'x_velocity [m/s]',
                                   'y_velocity [m/s]'])

# Plotting the diagram
#plt.figure(figsize=(12, 6))
#plt.subplot(4, 1, 1)
#plt.plot(total_sizes_contours, "o", label='Total Size Contours')
#plt.title('Total Size of Contours per frame')
#plt.xlabel('Frame Number')
#plt.ylabel('Size (px^2)')
# Annotate the plot with IDs at the corresponding frame on the x-axis where they are first detected
#for id, size in ids_and_sizes.items():
#    plt.text(first_detected_frame[id], size, str(id), fontsize=14, color='red')
#plt.legend()

#plt.subplot(4, 1, 2)
#plt.plot(num_contours_per_frame, label='Number of Contours')
#plt.title('Number of Contours per frame')
#plt.xlabel('Frame Number')
#plt.ylabel('Count')
#plt.legend()

#plt.subplot(4, 1, 3)
#tracker.plot_distances()

#plt.subplot(4, 1, 4)
#tracker.plot_total_distances()

#plt.figure(figsize=(12,6))
#plt.subplot(2, 1, 1)
#tracker.plot_x_movement()

#plt.subplot(2, 1, 2)
#tracker.plot_y_movement()

#plt.tight_layout()
#plt.show()