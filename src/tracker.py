import numpy as np

class TrafficCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.counting_dict = {"entering": 0, "leaving": 0}
        self.tracked_ids = {} # {id: last_y}

    def update_counts(self, boxes):
        if boxes.id is None:
            return

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for obj_id, box in zip(ids, xyxy):
            # Calculate centroid Y coordinate
            center_y = (box[1] + box[3]) / 2
            
            if obj_id in self.tracked_ids:
                prev_y = self.tracked_ids[obj_id]
                
                # Check for crossing (Top to Bottom = Entering)
                if prev_y < self.line_y and center_y >= self.line_y:
                    self.counting_dict["entering"] += 1
                # Check for crossing (Bottom to Top = Leaving)
                elif prev_y > self.line_y and center_y <= self.line_y:
                    self.counting_dict["leaving"] += 1
            
            self.tracked_ids[obj_id] = center_y

    def get_counts(self):
        return self.counting_dict