import math
import pandas as pd
import matplotlib.pyplot as plt

class EuclideanDistTracker:
    def __init__(self, vid_width):
        # Store the center positions of the objects
        self.center_points = {}
        self.center_points_1 = {}
        self.center_points_2 = {}
        self.center_points_3 = {}
        self.center_points_4 = {}
        self.center_points_5 = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0
        self.vid_width = vid_width
        self.max_lines = 100
        # Store the distances traveled between frames for each individual object
        self.distances_dict = {}
        self.total_distances_dict = {}
        self.dist_thresh = 70

    def update(self, objects_centers, frame_num):
        # Objects boxes and ids
        objects_ids = []

        # Get center point of new object
        for center in objects_centers:
            cx, cy = center
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < self.dist_thresh and same_object_detected == False:
                    self.center_points[id] = (cx, cy)
                    
                    print("prev-0: " + str(self.center_points))

                    new_dist = dist + self.total_distances_dict[id][-1][0]
                    self.total_distances_dict[id].append((new_dist, frame_num))

                    self.distances_dict[id].append((dist, frame_num))
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True

                    #print("----------------")
                    #print("prev-0: " + str(self.center_points) + ", frame: " + str(frame_num))
                    #print("Total dist dict: " + str(self.total_distances_dict))
                    #print("Distance: " + str(dist) + ", ID: " + str(id) + ", frame: " + str(frame_num))
                    #print("dict: " + str(self.distances_dict))

                    break
            
            for id, pt in self.center_points_1.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.dist_thresh and same_object_detected == False:
                    self.center_points[id] = (cx, cy)

                    print("prev-1: " + str(self.center_points))

                    new_dist = dist + self.total_distances_dict[id][-1][0]
                    self.total_distances_dict[id].append((new_dist, frame_num))


                    self.distances_dict[id].append((dist, frame_num))
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break
            
            for id, pt in self.center_points_2.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.dist_thresh and same_object_detected == False:
                    self.center_points[id] = (cx, cy)

                    print("prev-2: " + str(self.center_points))

                    new_dist = dist + self.total_distances_dict[id][-1][0]
                    self.total_distances_dict[id].append((new_dist, frame_num))


                    self.distances_dict[id].append((dist, frame_num))
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_3.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.dist_thresh and same_object_detected == False:
                    self.center_points[id] = (cx, cy)

                    new_dist = dist + self.total_distances_dict[id][-1][0]
                    self.total_distances_dict[id].append((new_dist, frame_num))

                    
                    print("prev-3: " + str(self.center_points))
                    self.distances_dict[id].append((dist, frame_num))
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_4.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.dist_thresh and same_object_detected == False:
                    self.center_points[id] = (cx, cy)

                    new_dist = dist + self.total_distances_dict[id][-1][0]
                    self.total_distances_dict[id].append((new_dist, frame_num))

                    print("prev-4: " + str(self.center_points))
                    self.distances_dict[id].append((dist, frame_num))
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_5.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.dist_thresh and same_object_detected == False:
                    self.center_points[id] = (cx, cy)

                    new_dist = dist + self.total_distances_dict[id][-1][0]
                    self.total_distances_dict[id].append((new_dist, frame_num))

                    print("prev-5: " + str(self.center_points))
                    self.distances_dict[id].append((dist, frame_num))
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False and cy > self.vid_width*2/3:
                self.center_points[self.id_count] = (cx, cy)
                objects_ids.append((cx, cy, self.id_count))
                self.distances_dict[self.id_count] = []
                self.total_distances_dict[self.id_count] = [(0, frame_num)]
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_center_id in objects_ids:
            cx, cy, object_id = obj_center_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionaries and remove unused IDs
        self.center_points_5 = self.center_points_4.copy()
        self.center_points_4 = self.center_points_3.copy()
        self.center_points_3 = self.center_points_2.copy()
        self.center_points_2 = self.center_points_1.copy()
        self.center_points_1 = self.center_points.copy()
        self.center_points = new_center_points.copy()
        return objects_ids
    

    def plot_distances(self):
        for key, values in self.distances_dict.items():
            x = [val[1] for val in values]  # Extracting x values
            y = [val[0] for val in values]  # Extracting y values
            plt.plot(x, y, "o", label=f'Object {key}')  # Plotting the line

        # Add labels and legend
        plt.xlabel('Frame')
        plt.ylabel('Euclidean Distance Traveled')
        plt.title('Euclidean Distance Traveled frame to frame')
        plt.legend()
        plt.grid(True)

    def plot_total_distances(self):
        for key, values in self.total_distances_dict.items():
            x = [val[1] for val in values]  # Extracting x values
            y = [val[0] for val in values]  # Extracting y values
            plt.plot(x, y, "o", label=f'Object {key}')  # Plotting the line

        # Add labels and legend
        plt.xlabel('Frame')
        plt.ylabel('Total Distace traveled in each frame')
        plt.title('Total Distances')
        plt.legend()
        plt.grid(True)