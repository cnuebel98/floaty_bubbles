import math

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
        #print(str(self.vid_width))

    def update(self, objects_centers):
        # Objects boxes and ids
        objects_ids = []

        # Get center point of new object
        for center in objects_centers:
            cx, cy = center
            
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_1.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break
            
            for id, pt in self.center_points_2.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_3.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_4.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break

            for id, pt in self.center_points_5.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_ids.append((cx, cy, id))
                    same_object_detected = True
                    break


            # New object is detected we assign the ID to that object
            if same_object_detected is False and cy > self.vid_width*2/3:
                self.center_points[self.id_count] = (cx, cy)
                objects_ids.append((cx, cy, self.id_count))
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_center_id in objects_ids:
            cx, cy, object_id = obj_center_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points_5 = self.center_points_4.copy()
        self.center_points_4 = self.center_points_3.copy()
        self.center_points_3 = self.center_points_2.copy()
        self.center_points_2 = self.center_points_1.copy()
        self.center_points_1 = self.center_points.copy()
        self.center_points = new_center_points.copy()
        return objects_ids