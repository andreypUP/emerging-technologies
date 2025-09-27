import cv2
import numpy as np
import math
import os

# ---------- helper functions ----------

def circularity(contour):
    """
    Compute how close a shape is to a perfect circle.
    Formula: 4π * Area / Perimeter^2
    - 1.0 = perfect circle
    - closer to 0 = irregular shape
    """
    per = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if per == 0:
        return 0
    return 4 * math.pi * area / (per * per)

def aspect_ratio(contour):
    """
    Compute the width/height ratio of the bounding box.
    Useful for filtering out non-circular shapes (too elongated).
    """
    x,y,w,h = cv2.boundingRect(contour)
    return w/float(h) if h>0 else 1

def estimate_background_lab(image_bgr, margin=20):
    """
    Estimate the background color by sampling the 4 corners of the image.
    Convert them to LAB color space and average them.
    LAB is used because it represents color differences more naturally.
    """
    h, w = image_bgr.shape[:2]
    samples = []
    # Top-left, top-right, bottom-left, bottom-right patches
    samples.extend(image_bgr[0:margin, 0:margin].reshape(-1, 3))
    samples.extend(image_bgr[0:margin, w-margin:w].reshape(-1, 3))
    samples.extend(image_bgr[h-margin:h, 0:margin].reshape(-1, 3))
    samples.extend(image_bgr[h-margin:h, w-margin:w].reshape(-1, 3))
    samples = np.array(samples, dtype=np.uint8)
    samples_lab = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    return np.mean(samples_lab.reshape(-1, 3), axis=0)

def color_distance_mask_lab(image_bgr, bg_lab, thresh=35):
    """
    Build a binary mask separating coins from background:
    - Convert image to LAB
    - Compute distance of each pixel from estimated background color
    - Threshold: keep pixels farther than 'thresh' from background
    - Morphological close + open to clean noise
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = np.sqrt(
        (lab[:, :, 0] - bg_lab[0]) ** 2 +
        (lab[:, :, 1] - bg_lab[1]) ** 2 +
        (lab[:, :, 2] - bg_lab[2]) ** 2
    )
    mask = (diff > thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def watershed_segments_from_mask(mask, fg_thresh=0.4):
    """
    Use Watershed segmentation to separate touching coins.
    - Compute distance transform
    - Detect sure foreground (coin centers) and background
    - Apply watershed to split coins
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_thresh * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(mask, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_bgr, markers)
    return markers

def extract_contours_from_markers(markers):
    """
    Extract coin-shaped contours from watershed segmentation markers.
    Each unique label corresponds to a region.
    """
    contours = []
    for label in np.unique(markers):
        if label <= 1:  # skip background and border
            continue
        mask = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(cnts)
    return contours

# ---------- detection pipeline ----------
def detect_coins_pipeline(image_bgr, params):
    """
    Complete coin detection pipeline:
    1. Background subtraction
    2. Watershed segmentation
    3. Shape filtering (circularity, aspect ratio, area)
    4. Deduplication of overlapping detections
    5. Fallback with HoughCircles if too few detections
    6. Draw final detections
    """
    h, w = image_bgr.shape[:2]
    minR = max(8, int(min(h, w) * 0.015))  # min coin radius
    maxR = int(min(h, w) * 0.09)           # max coin radius

    # Step 1: background mask
    bg_lab = estimate_background_lab(image_bgr)
    mask = color_distance_mask_lab(image_bgr, bg_lab, thresh=params["bg_thresh"])

    # Step 2: watershed segmentation
    markers = watershed_segments_from_mask(mask, fg_thresh=params["fg_thresh"])
    seg_contours = extract_contours_from_markers(markers)

    # Step 3: filter contours
    candidates = []
    min_area = math.pi * (minR * 0.5) ** 2
    max_area = math.pi * (maxR * 1.3) ** 2
    for c in seg_contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        circ = circularity(c)
        ar = aspect_ratio(c)
        # filter out shapes too elongated or not circular enough
        if circ < params["min_circ"] or ar < params["min_ar"] or ar > params["max_ar"]:
            continue

        # estimate coin radius
        (x, y), r1 = cv2.minEnclosingCircle(c)
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            r2 = (ellipse[1][0] + ellipse[1][1]) / 4.0
        else:
            r2 = r1
        r = 0.5 * r1 + 0.5 * r2
        if circ > 0.85:
            r *= 1.05  # slightly enlarge for near-perfect circles

        x, y, r = int(round(x)), int(round(y)), int(round(r))
        if r < minR//2 or r > maxR*1.3:
            continue
        candidates.append((x, y, r, area, circ))

    # Step 4: deduplicate overlapping candidates
    candidates_sorted = sorted(candidates, key=lambda t: t[3], reverse=True)
    final, taken = [], []
    for x, y, r, area, circ in candidates_sorted:
        if all(math.hypot(x - fx, y - fy) > max(r, fr) * params["dedup_dist"] for fx, fy, fr in taken):
            final.append((x, y, r))
            taken.append((x, y, r))

    # Step 5: fallback with HoughCircles if too few detections
    if len(final) < 4:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=minR*1.5, param1=100, param2=params["hough_param2"],
            minRadius=minR, maxRadius=maxR
        )
        if circles is not None:
            for (x, y, r) in np.round(circles[0]).astype(int):
                if all(math.hypot(x - fx, y - fy) > max(r, fr) * params["dedup_dist"] for fx, fy, fr in taken):
                    final.append((x, y, r))
                    taken.append((x, y, r))

    # Step 6: draw detections
    out = image_bgr.copy()
    for (x, y, r) in final:
        cv2.circle(out, (x, y), r, (0, 255, 0), 2)   # draw coin boundary
        cv2.circle(out, (x, y), 2, (0, 0, 255), 3)   # draw coin center

    cv2.putText(out, f"Coins: {len(final)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return out, final

# ---------- main ----------
if __name__ == "__main__":
    # Parameters for tuning detection
    params = {
        "bg_thresh": 35,     # LAB distance from background
        "fg_thresh": 0.40,   # watershed foreground fraction
        "min_circ": 0.55,    # min circularity
        "min_ar": 0.7,       # min aspect ratio
        "max_ar": 1.3,       # max aspect ratio
        "dedup_dist": 0.65,  # deduplication distance factor
        "hough_param2": 35,  # HoughCircles sensitivity
    }

    # Input images
    img_dir = "images"
    image_files = ["coins (1).png","coins (2).jpg","coins (3).jpg",
                   "coins (4).jpg","coins (5).jpg","coins (6).jpg","coins (7).jpg"]

    results = []
    for fname in image_files:
        path = os.path.join(img_dir, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"Error loading {path}")
            continue
        out, coins = detect_coins_pipeline(image, params)
        print(f"{fname}: {len(coins)} coins detected")
        results.append(out)

    # Create a collage of results
    target_size = (600, 600)
    results_resized = [cv2.resize(r, target_size) for r in results]
    while len(results_resized) < 8:
        results_resized.append(np.zeros((600, 600, 3), dtype=np.uint8))
    row1 = np.hstack(results_resized[0:4])
    row2 = np.hstack(results_resized[4:8])
    collage = np.vstack([row1, row2])

    cv2.imshow("Detected Coins (All Images)", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
