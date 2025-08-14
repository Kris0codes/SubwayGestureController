import cv2
import time
import math
import numpy as np
from collections import deque
import pyautogui
import torch
import os

# Import the PalmDetector from the repo
from BlazePalm.ML.blazepalm import PalmDetector

#  PyAutoGUI settings
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

#  Utilities for gesture control
class Debounce:
    def __init__(self, cooldown=0.6):
        self.cooldown = cooldown
        self.last_time = {}

    def ready(self, key):
        now = time.time()
        if self.last_time.get(key, 0) + self.cooldown <= now:
            self.last_time[key] = now
            return True
        return False

def press(key, debouncer, label, frame, color=(0, 255, 0)):
    if debouncer.ready(key):
        pyautogui.press(key)
        cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

def preprocess_for_palm(frame_bgr):
    """
    PalmDetector.predict_on_batch expects (B,3,256,256) and will internally
    scale to [-1,1]. We just need to make a torch tensor (uint8 or float).
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0)  # (1,3,256,256)
    return x  # dtype=uint8 is fine; predict_on_batch will x.float() inside

def first_detection_to_box(dets_tensor):
    """
    dets_tensor: shape (N, 19). Coords are:
      [ymin, xmin, ymax, xmax, 14 keypoint coords..., score]
    Returns (xmin, ymin, xmax, ymax, score) in normalized [0..1] coords.
    """
    if dets_tensor is None or len(dets_tensor) == 0:
        return None
    d = dets_tensor[0]  
    ymin, xmin, ymax, xmax = d[0].item(), d[1].item(), d[2].item(), d[3].item()
    score = d[18].item()
    return xmin, ymin, xmax, ymax, score

def main():
    #Init PalmDetector on CPU 
    palm = PalmDetector()
    # Load weights (required)
    # The weights file should be in the same directory as this script or in BlazePalm/ML/
    if os.path.exists("palmdetector.pth"):
        palm.load_weights("palmdetector.pth")
    elif os.path.exists(os.path.join("BlazePalm", "ML", "palmdetector.pth")):
        palm.load_weights(os.path.join("BlazePalm", "ML", "palmdetector.pth"))
    else:
        print("[WARN] palmdetector.pth not found. If you see bad results, place it next to this script.")

    # Load anchors (required)
    if os.path.exists(os.path.join("BlazePalm", "ML", "anchors.npy")):
        palm.load_anchors(os.path.join("BlazePalm", "ML", "anchors.npy"))
    elif os.path.exists("anchors.npy"):
        palm.load_anchors("anchors.npy")
    else:
        raise FileNotFoundError("anchors.npy not found. Put it at BlazePalm/ML/anchors.npy or next to this script.")

    palm.eval()  # important

    #Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    # Gesture state
    debouncer = Debounce(cooldown=0.6)

    # Keep short history of box centers to estimate swipe velocity
    x_hist = deque(maxlen=6)
    t_hist = deque(maxlen=6)

    # Tunables
    SWIPE_VEL = 1.0         # normalized units / second
    JUMP_Y = 0.35           # center y < 0.35 => jump
    CROUCH_Y = 0.65         # center y > 0.65 => crouch
    SCORE_THRESH = 0.6      # detection confidence threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        #Detect palm 
        palm_in = preprocess_for_palm(frame)  # CPU tensor, shape (1,3,256,256)
        with torch.no_grad():
            det_list = palm.predict_on_batch(palm_in)  # list length = batch size (1)
            # det_list[0] is either empty (no dets) or a tensor (N, 19)
            dets = det_list[0] if len(det_list) > 0 else None

        if dets is not None and len(dets) > 0:
            # Convert first detection
            box = first_detection_to_box(dets)
            if box is not None:
                xmin, ymin, xmax, ymax, score = box
                if score >= SCORE_THRESH:
                    # Draw box
                    ix0, iy0 = int(xmin * w), int(ymin * h)
                    ix1, iy1 = int(xmax * w), int(ymax * h)
                    cv2.rectangle(frame, (ix0, iy0), (ix1, iy1), (0, 255, 0), 2)
                    cv2.putText(frame, f"score:{score:.2f}", (ix0, max(iy0-6, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Box center in normalized coords
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0

                    # Update velocity history
                    now = time.time()
                    x_hist.append(cx)
                    t_hist.append(now)

                    # Gestures from box position
                    # Jump (hand up)
                    if cy < JUMP_Y:
                        press('up', debouncer, 'JUMP ↑', frame)
                    # Crouch (hand down)
                    elif cy > CROUCH_Y:
                        press('down', debouncer, 'CROUCH ↓', frame)

                    # Swipe left/right from horizontal velocity
                    if len(x_hist) >= 2:
                        dx = x_hist[-1] - x_hist[0]
                        dt = t_hist[-1] - t_hist[0]
                        if dt > 0:
                            vx = dx / dt  # normalized units per second
                            if vx < -SWIPE_VEL:
                                press('left', debouncer, 'LEFT ←', frame, (255, 255, 0))
                                x_hist.clear(); t_hist.clear()
                            elif vx > SWIPE_VEL:
                                press('right', debouncer, 'RIGHT →', frame, (255, 255, 0))
                                x_hist.clear(); t_hist.clear()

        # Visual guides
        cv2.line(frame, (0, int(JUMP_Y * h)), (w, int(JUMP_Y * h)), (160, 160, 160), 1)
        cv2.line(frame, (0, int(CROUCH_Y * h)), (w, int(CROUCH_Y * h)), (160, 160, 160), 1)
        cv2.putText(frame, "q: quit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Subway Surfers - Palm Gesture Control (bbox-based)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
