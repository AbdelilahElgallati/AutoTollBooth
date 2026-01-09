import cv2
from .detector import VehicleDetector
from .tracker import TrafficCounter

class VideoProcessor:
    def __init__(self, model_path, line_y_ratio=0.6):
        self.detector = VehicleDetector(model_path)
        self.line_y_ratio = line_y_ratio
        self.counter = None

    def process_frame(self, frame, conf):
        h, w, _ = frame.shape
        line_y = int(h * self.line_y_ratio)
        
        if self.counter is None:
            self.counter = TrafficCounter(line_y)

        # 1. Detect and Track
        results = self.detector.detect_and_track(frame, conf)
        
        # 2. Update Counter
        if results.boxes is not None:
            self.counter.update_counts(results.boxes)

        # 3. Annotate Frame
        annotated_frame = results.plot()
        cv2.line(annotated_frame, (0, line_y), (w, line_y), (0, 255, 255), 3)
        
        return annotated_frame, self.counter.get_counts()