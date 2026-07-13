import argparse
import cv2
import numpy as np
import napari

# Order the 4 coords geometrically: top left, top right, bottom right, bottom left
def order_points(pts: np.ndarray):
    # 1. Find the centroid (center point) of the 4 coordinates
    center = np.mean(pts, axis=0)
    
    # 2. Calculate the angle of each point relative to the center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # 3. Sort the points by angle to create a clockwise cyclic order.
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # 4. Calculate Euclidean distance to the image origin (0,0) to find the Top-Left point.
    # This is far more robust to severe perspective distortion than the x+y sum.
    dists = np.linalg.norm(sorted_pts, axis=1)
    tl_index = np.argmin(dists)
    
    # 5. Shift the array circularly so the Top-Left point is first.
    ordered_rect = np.roll(sorted_pts, -tl_index, axis=0)
    
    return ordered_rect

def main():
    parser = argparse.ArgumentParser(description="Whiteboard perspective correction using ArUco markers.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image")
    parser.add_argument("-w", "--max-width", type=int, default=1920, help="Maximum width of the output image")
    parser.add_argument("-t", "--max-height", type=int, default=1080, help="Maximum height of the output image")
    parser.add_argument("-ao", "--aruco-order", action="store_true", help="Sort corners using ArUco IDs (Lowest ID = TL, 2nd = TR, 3rd = BR, Highest = BL)")
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

    # 4. Extract the center points and their corresponding IDs
    centers = []
    detected_ids = []
    
    # Flatten ids to handle different OpenCV version return shapes safely
    flat_ids = ids.flatten() 
    
    for i in range(4):
        marker_id = flat_ids[i]
        marker_corners = corners[i][0]
        cX = np.mean(marker_corners[:, 0])
        cY = np.mean(marker_corners[:, 1])
        centers.append([cX, cY])
        detected_ids.append(marker_id)

    centers = np.array(centers, dtype="float32")
    
    # 5. Order the centers (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    if args.aruco_order:
        # Sort by marker ID ascending
        sorted_pairs = sorted(zip(detected_ids, centers), key=lambda x: x[0])
        ordered_centers = np.array([pair[1] for pair in sorted_pairs], dtype="float32")
    else:
        # Fallback to geometric sorting
        ordered_centers = order_points(centers)
        
    (tl, tr, br, bl) = ordered_centers

    # --- Calculate True Aspect Ratio and Fit to Maximum Bounds ---
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

    # 6. Define the destination points for the flattened whiteboard
    dst_points = np.array([
        [0, 0],
        [out_width - 1, 0],
        [out_width - 1, out_height - 1],
        [0, out_height - 1]
    ], dtype="float32")

    # 7. Compute the homography matrix and warp
    H, status = cv2.findHomography(ordered_centers, dst_points)
    warped = cv2.warpPerspective(image, H, (out_width, out_height))

    # --- Visualization ---
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    for pt in ordered_centers:
        cv2.circle(image, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)

    print(f"Output resolution: {out_width}x{out_height}")
    print(f"Sorting method: {'ArUco IDs' if args.aruco_order else 'Geometric (Euclidean)'}")

    # 1. Open Window 1 for the original image
    viewer1 = napari.Viewer(title="Detected Markers")
    viewer1.add_image(image, rgb=True)

    # 2. Open Window 2 for the warped image
    viewer2 = napari.Viewer(title="Corrected Whiteboard")
    viewer2.add_image(warped, rgb=True)

    # 4. Start the event loop (replaces cv2.waitKey)
    print("Close the viewers to exit...")
    napari.run()

if __name__ == "__main__":
    main()