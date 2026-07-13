import argparse
import cv2
import numpy as np

# Order the 4 coords: top left, top right, bottom right, bottom left
def order_points(pts: np.ndarray):
    # 1. Find the centroid (center point) of the 4 coordinates
    center = np.mean(pts, axis=0)
    
    # 2. Calculate the angle of each point relative to the center
    # np.arctan2(y, x) returns angles from -pi to pi.
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # 3. Sort the points by angle. 
    # Because OpenCV Y-coordinates go down, this creates a clockwise cyclic order.
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # 4. We now have a guaranteed valid quadrilateral, but we don't know which 
    # point is "Top-Left". The point with the smallest (x + y) sum is visually
    # the closest to the origin (0,0) in the image.
    sums = sorted_pts.sum(axis=1)
    tl_index = np.argmin(sums)
    
    # 5. Shift the array circularly so the Top-Left point is first.
    # The cyclic order remains intact, resulting in: TL, TR, BR, BL
    ordered_rect = np.roll(sorted_pts, -tl_index, axis=0)
    
    return ordered_rect

def order_points2(pts: np.ndarray):
    pts_list = pts.tolist()
    for x, y in pts_list:
        top = 0
        left = 0
        for i, j in pts_list:
            if x < i:
                left += 1
            elif x > i:
                left -= 1
            if y < j:
                top += 1
            elif y > j:
                top -= 1
        
        if top > 0 and left > 0:
            tl = (x, y)
        if top > 0 and left < 0:
            tr = (x, y)
        if top < 0 and left > 0:
            bl = (x, y)
        if top < 0 and left < 0:
            br = (x, y)

    out_list = [tl, tr, br, bl]
    return

def main():
    parser = argparse.ArgumentParser(description="Whiteboard perspective correction using ArUco markers.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image")
    parser.add_argument("-w", "--max-width", type=int, default=1920, help="Maximum width of the output image")
    parser.add_argument("-t", "--max-height", type=int, default=1080, help="Maximum height of the output image")
    args = parser.parse_args()

    # 1. Load the image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image from {args.input}")
        return

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Configure the ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 3. Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        print(f"Error: Found {0 if ids is None else len(ids)} markers. Exactly 4 are required.")
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            cv2.imshow("Found Markers", image)
            cv2.waitKey(0)
        return

    # 4. Extract the center points of the first 4 detected markers
    centers = []
    for i in range(4):
        marker_corners = corners[i][0]
        cX = np.mean(marker_corners[:, 0])
        cY = np.mean(marker_corners[:, 1])
        centers.append([cX, cY])

    centers = np.array(centers, dtype="float32")
    
    # Order the centers (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    ordered_centers = order_points(centers)
    (tl, tr, br, bl) = ordered_centers

    print(centers)
    print(ordered_centers)
    input()

    # --- NEW: Calculate True Aspect Ratio and Fit to Maximum Bounds ---
    # Compute the width of the new image (max distance between top-right/top-left and bottom-right/bottom-left)
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    target_width = max(int(width_top), int(width_bottom))

    # Compute the height of the new image (max distance between top-right/bottom-right and top-left/bottom-left)
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    target_height = max(int(height_right), int(height_left))

    # Scale to fit inside max bounds while maintaining aspect ratio
    scale_w = args.max_width / target_width
    scale_h = args.max_height / target_height
    scale = min(scale_w, scale_h)  # Choose the limiting scale factor

    out_width = int(target_width * scale)
    out_height = int(target_height * scale)

    # 5. Define the destination points for the flattened whiteboard using dynamically calculated sizes
    dst_points = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype="float32")

    # 6. Compute the homography matrix
    H, status = cv2.findHomography(ordered_centers, dst_points)

    # 7. Warp the perspective to get the top-down view (using out_width and out_height)
    warped = cv2.warpPerspective(image, H, (out_width, out_height))

    # --- Visualization ---
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    for pt in ordered_centers:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)

    # Resize input image just to fit on a standard monitor
    scale_percent = 50 
    img_h = int(image.shape[0] * scale_percent / 100)
    img_w = int(image.shape[1] * scale_percent / 100)
    resized_original = cv2.resize(image, (img_w, img_h))

    cv2.imshow("Detected Markers", resized_original)
    cv2.imshow("Corrected Whiteboard", warped)
    print(f"Output resolution: {out_width}x{out_height}")
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()