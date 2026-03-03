import os
# Force Qt to use X11/XWayland to avoid crashes on Wayland-only systems
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time

# Path to the DNN face detection model (ResNet-SSD, much more accurate than Haar cascades)
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTOTXT = os.path.join(_DIR, "models", "deploy.prototxt")
_CAFFEMODEL = os.path.join(_DIR, "models", "res10_300x300_ssd.caffemodel")

class VideoHandler:
    def __init__(self):
        # Use V4L2 backend explicitly — the default backend stalls on first read
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video camera. Is it plugged in and not in use?")

        # Set YUV format (the only one this camera supports) and resolution
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Warm up: camera hardware needs ~1.5s to start streaming after open
        print("Warming up camera...")
        import time as _t
        _t.sleep(1.5)
        # Flush stale frames from the buffer
        for _ in range(10):
            self.cap.grab()
        print("Camera ready!")

        # Load DNN face detector (ResNet SSD - Caffe)
        print("Loading face detection model (DNN)...")
        if not os.path.exists(_PROTOTXT) or not os.path.exists(_CAFFEMODEL):
            raise FileNotFoundError(
                f"Face detection model files not found.\n"
                f"Expected:\n  {_PROTOTXT}\n  {_CAFFEMODEL}"
            )
        self.net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _CAFFEMODEL)
        print("Face detection model loaded!")

        self.CONFIDENCE_THRESHOLD = 0.5  # Only detect faces with >50% confidence
        self._show_window = True

    def release(self):
        """Release the camera and close any open windows."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    def _detect_faces(self, frame):
        """Run DNN face detection on a frame. Returns list of bounding boxes."""
        h, w = frame.shape[:2]
        # Resize to 300x300 and normalize for the DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)  # Mean subtraction for BGR
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1, confidence))
        return faces

    def _read_frame(self):
        """Read a single fresh frame from the camera."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _show_frame(self, frame, faces, status_text=""):
        """Draw detection boxes and status text, then show window."""
        if not self._show_window:
            return
        for (x, y, w, h, conf) in faces:
            label = f"Face {conf:.0%}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)
        if status_text:
            cv2.putText(frame, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("John AI - Vision [Q to quit]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt("User closed the camera window.")

    def wait_for_person(self, required_consecutive=4):
        """
        Blocks until a face is detected for `required_consecutive` frames in a row.
        Returns True when a person is found.
        """
        print("\n[Vision] Watching for a person...", flush=True)
        detect_count = 0

        while True:
            frame = self._read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            faces = self._detect_faces(frame)
            self._show_frame(frame, faces, "Waiting for person...")

            if faces:
                detect_count += 1
            else:
                detect_count = 0

            if detect_count >= required_consecutive:
                print("[Vision] Person detected! Activating...", flush=True)
                return True

    def is_person_in_frame(self, check_frames=6):
        """
        Quick check: is there still a person visible?
        Looks at `check_frames` and returns True if a face is found in ANY of them.
        """
        found = 0
        for _ in range(check_frames):
            frame = self._read_frame()
            if frame is None:
                continue
            faces = self._detect_faces(frame)
            self._show_frame(frame, faces, "Listening...")
            if faces:
                found += 1
            time.sleep(0.04)
        # If face detected in at least 2 out of check_frames, consider them present
        return found >= 2
