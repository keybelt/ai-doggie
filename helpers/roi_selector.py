import cv2
import sys
import time
from src.capture import start_capture
from AppKit import NSApplication


def main():
    _ = NSApplication.sharedApplication()

    print("[ROI] Connecting to Vision Engine...")
    engine = start_capture()
    engine.paused = False

    print("[ROI] Waiting for a frame from Geometry Dash...")
    frame = None

    start_wait = time.time()
    while time.time() - start_wait < 5.0:
        if not engine.ready_queue.empty():
            raw_data, _ = engine.ready_queue.get()
            frame = raw_data[:, :, :3].copy()
            break
        time.sleep(0.1)

    if frame is None:
        print("\n[ERROR] Could not grab a frame. Is Geometry Dash open?")
        sys.exit(1)

    print("\n[INSTRUCTIONS]")
    print("1. A window will pop up showing the game feed.")
    print("2. Click and drag a box around the '0%' or 'Attempt' text tightly.")
    print("3. Press SPACE or ENTER to confirm the selection.")
    print("4. Press 'c' to cancel.")

    roi = cv2.selectROI("Select 0% Label", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        print("\n[ROI] Selection cancelled.")
        sys.exit()

    crop = frame[y:y + h, x:x + w]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("0%.png", gray_crop)

    print(f"\n✅ SUCCESS! Reference image saved as '0%.png'.")
    print("-" * 40)
    print(f"Paste this line into env.py:")
    print(f"\033[92mself.attempt_roi = ({x}, {y}, {w}, {h})\033[0m")
    print("-" * 40)


if __name__ == "__main__":
    main()