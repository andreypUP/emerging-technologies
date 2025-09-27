import numpy as np
import argparse
import time
import cv2

# define HSV color ranges
def color_range(color):
    if color == 'blue':
        lower1 = np.array([85, 40, 40])
        upper1 = np.array([130, 255, 255])
        return [(lower1, upper1)]
    elif color == 'green':
        return [(np.array([40, 70, 70]), np.array([80, 255, 255]))]
    elif color == 'red':
        # Red has two ranges in HSV
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
        return [(lower1, upper1), (lower2, upper2)]
    else:
        raise ValueError("Color must be blue, green, or red")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the (optional) video file")
    ap.add_argument("-c", "--color", default="blue",
                    help="color option: blue, green, or red (default=blue)")
    args = vars(ap.parse_args())

    video = args["video"] if args["video"] is not None else 0
    camera = cv2.VideoCapture(video)

    # make windows resizable
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)

    # get screen resolution (change if needed)
    screen_width = 1280
    screen_height = 720

    # resize windows to half screen width each
    cv2.resizeWindow("Tracking", screen_width // 2, screen_height)
    cv2.resizeWindow("Binary", screen_width // 2, screen_height)

    # move windows side by side
    cv2.moveWindow("Tracking", 0, 0)
    cv2.moveWindow("Binary", screen_width // 2, 0)

    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # build mask for the selected color
        masks = [cv2.inRange(hsv, lower, upper) for (lower, upper) in color_range(args["color"])]
        color_mask = masks[0]
        if len(masks) > 1:
            color_mask = cv2.bitwise_or(masks[0], masks[1])

        color_mask = cv2.GaussianBlur(color_mask, (3, 3), 0)

        cnts, _ = cv2.findContours(color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            cnt = max(cnts, key=cv2.contourArea)
            rect = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

        # show both windows
        cv2.imshow("Tracking", frame)     # normal camera view
        cv2.imshow("Binary", color_mask)  # binary mask view

        time.sleep(0.025)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
