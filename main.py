import argparse
import cv2
import numpy as np

def order_points(pts):
    """
    Orders a list of 4 coordinates consistently:
    Top-Left, Top-Right, Bottom-Right, Bottom-Left
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Compute the difference between the points
    # the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def main():
    parser = argparse.ArgumentParser(description="Whiteboard perspective correction using ArUco markers.")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    parser.add_argument("-w", "--width", type=int, default=1920, help="Width of the output corrected image")
    parser.add_argument("-t", "--height", type=int, default=1080, help="Height of the output corrected image")
    args = parser.parse_args()

    # 1. Load the image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from {args.image}")
        return

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Configure the ArUco detector
    # DICT_4X4_50 is standard for simple, easily detectable markers
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
        # corners[i] has shape (1, 4, 2)
        marker_corners = corners[i][0]
        cX = np.mean(marker_corners[:, 0])
        cY = np.mean(marker_corners[:, 1])
        centers.append([cX, cY])

    centers = np.array(centers, dtype="float32")
    
    # Order the centers (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    ordered_centers = order_points(centers)

    # 5. Define the destination points for the flattened whiteboard
    dst_points = np.array([
        [0, 0],
        [args.width - 1, 0],
        [args.width - 1, args.height - 1],
        [0, args.height - 1]
    ], dtype="float32")

    # 6. Compute the homography matrix
    H, status = cv2.findHomography(ordered_centers, dst_points)

    # 7. Warp the perspective to get the top-down view
    warped = cv2.warpPerspective(image, H, (args.width, args.height))

    # --- Visualization ---
    # Draw original markers and calculated centers on the input image
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
    print("Press any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()