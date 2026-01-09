import os
import shutil
from ultralytics import YOLO


def _ensure_model_in_models_dir(model_path: str) -> str:
    """Ensure the requested model file exists under the project's `models/` folder.

    If `model_path` is a filename (e.g. 'yolov8n.pt') this will check
    '<project_root>/models/yolov8n.pt' and, if missing, attempt to trigger
    Ultralytics to download the model and move it into the models folder.

    Returns the filesystem path to the model to load.
    """
    # Project root is one level up from src/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_filename = os.path.basename(model_path)
    target_path = os.path.join(models_dir, model_filename)

    # If an explicit path was provided and exists, use it directly
    if os.path.isabs(model_path) or os.path.dirname(model_path):
        if os.path.exists(model_path):
            return model_path

    # If already present in models/, return it
    if os.path.exists(target_path):
        return target_path

    # Try to see if the model exists at given relative path (cwd etc.)
    if os.path.exists(model_path):
        try:
            shutil.copy2(model_path, target_path)
            return target_path
        except Exception:
            pass

    # Otherwise, attempt to let Ultralytics download the model. In many
    # environments YOLO(<name>) will download a file named <name> into the
    # current working directory; if that happens we move it to models/.
    try:
        print(f"[VehicleDetector] Model not found at {target_path}. Attempting to download '{model_path}'...")
        # Instantiate YOLO which may download the weights into Ultralytics cache
        m = YOLO(model_path)

        # Common place Ultralytics places the downloaded file is the current
        # working directory with the same filename. Try to move it to models/.
        cwd_candidate = os.path.join(os.getcwd(), model_filename)
        if os.path.exists(cwd_candidate):
            try:
                shutil.move(cwd_candidate, target_path)
                print(f"[VehicleDetector] Moved downloaded model to {target_path}")
                return target_path
            except Exception:
                pass

        # If we couldn't move it, but the YOLO object was created, prefer to
        # use the model instance's internal path if available. Fall back to
        # loading via the original model_path (Ultralytics will handle cache).
        print(f"[VehicleDetector] Could not move downloaded file to models/. Using Ultralytics cached model for '{model_path}'.")
        return model_path
    except Exception as e:
        # If download failed, raise a helpful error
        raise RuntimeError(f"Failed to ensure model '{model_path}' is available: {e}")


def ensure_model(model_path: str) -> str:
    """Public wrapper around the internal helper.

    Use this from other modules to ensure a model exists in `models/`.
    Returns the path that should be used to load the model.
    """
    return _ensure_model_in_models_dir(model_path)


class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Ensure model is present in models/ if possible, then load it
        model_to_load = _ensure_model_in_models_dir(model_path)
        self.model = YOLO(model_to_load)

        # Class indices for common vehicles in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def detect_and_track(self, frame, conf=0.25):
        # Run tracking on the frame
        results = self.model.track(
            frame,
            persist=True,
            classes=self.vehicle_classes,
            conf=conf,
            verbose=False,
        )
        return results[0]