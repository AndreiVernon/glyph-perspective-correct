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

def process_image(image, detector, args):
    """
    Processes a single frame.
    Returns: (warped_image, annotated_image, success_boolean)
    """
    annotated = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    num_markers = 0 if ids is None else len(ids)

    if num_markers not in [1, 2, 4]:
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        return None, annotated, False

    flat_ids = ids.flatten()

    if num_markers == 4:
        # --- 4-GLYPH MODE: Crop to the 4 corners ---
        centers = []
        detected_ids = []
        
        for i in range(4):
            marker_id = flat_ids[i]
            marker_corners = corners[i][0]
            cX = np.mean(marker_corners[:, 0])
            cY = np.mean(marker_corners[:, 1])
            centers.append([cX, cY])
            detected_ids.append(marker_id)

        centers = np.array(centers, dtype="float32")
        
        if args.aruco_order:
            sorted_pairs = sorted(zip(detected_ids, centers), key=lambda x: x[0])
            ordered_centers = np.array([pair[1] for pair in sorted_pairs], dtype="float32")
        else:
            ordered_centers = order_points(centers)
            
        (tl, tr, br, bl) = ordered_centers

        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        target_width = max(int(width_top), int(width_bottom))

        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        target_height = max(int(height_right), int(height_left))

        scale_w = args.max_width / target_width
        scale_h = args.max_height / target_height
        scale = min(scale_w, scale_h)

        out_width = int(target_width * scale)
        out_height = int(target_height * scale)

        dst_points = np.array([
            [0, 0],
            [out_width - 1, 0],
            [out_width - 1, out_height - 1],
            [0, out_height - 1]
        ], dtype="float32")

        H, status = cv2.findHomography(ordered_centers, dst_points)
        warped = cv2.warpPerspective(image, H, (out_width, out_height))

        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for pt in ordered_centers:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)

        return warped, annotated, True

    elif num_markers == 2:
        # --- 2-GLYPH MODE: Average perspective, crop to rectangle defined by centers ---
        m1_corners = corners[0][0]
        m2_corners = corners[1][0]
        m1_center = np.mean(m1_corners, axis=0)
        m2_center = np.mean(m2_corners, axis=0)

        offsets1 = m1_corners - m1_center
        offsets2 = m2_corners - m2_center
        avg_offsets = (offsets1 + offsets2) / 2.0
        virtual_corners = ((m1_center + m2_center) / 2.0) + avg_offsets

        S = 100.0
        dst_marker_pts = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype="float32")
        H_temp, _ = cv2.findHomography(virtual_corners, dst_marker_pts)

        centers_to_transform = np.array([m1_center, m2_center], dtype="float32").reshape(-1, 1, 2)
        transformed_centers = cv2.perspectiveTransform(centers_to_transform, H_temp).squeeze()

        c1_t, c2_t = transformed_centers
        min_x, max_x = min(c1_t[0], c2_t[0]), max(c1_t[0], c2_t[0])
        min_y, max_y = min(c1_t[1], c2_t[1]), max(c1_t[1], c2_t[1])

        target_width = max_x - min_x
        target_height = max_y - min_y

        if target_width < 1 or target_height < 1:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            return None, annotated, False

        scale_w = args.max_width / target_width
        scale_h = args.max_height / target_height
        scale = min(scale_w, scale_h)

        out_width = int(target_width * scale)
        out_height = int(target_height * scale)

        dst_marker_pts_adjusted = (dst_marker_pts - np.array([min_x, min_y], dtype="float32")) * scale
        
        H, _ = cv2.findHomography(virtual_corners, dst_marker_pts_adjusted)
        warped = cv2.warpPerspective(image, H, (out_width, out_height))

        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        cv2.circle(annotated, (int(m1_center[0]), int(m1_center[1])), 10, (0, 0, 255), -1)
        cv2.circle(annotated, (int(m2_center[0]), int(m2_center[1])), 10, (0, 0, 255), -1)

        return warped, annotated, True

    elif num_markers == 1:
        # --- 1-GLYPH MODE: Correct perspective, do not crop ---
        m_corners = corners[0][0]
        
        S = 100.0
        dst_marker_pts = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype="float32")
        H_temp, _ = cv2.findHomography(m_corners, dst_marker_pts)

        h_orig, w_orig = image.shape[:2]
        img_corners = np.array([
            [0, 0], [w_orig - 1, 0], [w_orig - 1, h_orig - 1], [0, h_orig - 1]
        ], dtype="float32").reshape(-1, 1, 2)
        transformed_img_corners = cv2.perspectiveTransform(img_corners, H_temp).squeeze()

        min_x = np.min(transformed_img_corners[:, 0])
        max_x = np.max(transformed_img_corners[:, 0])
        min_y = np.min(transformed_img_corners[:, 1])
        max_y = np.max(transformed_img_corners[:, 1])

        target_width = max_x - min_x
        target_height = max_y - min_y

        scale_w = args.max_width / target_width
        scale_h = args.max_height / target_height
        scale = min(scale_w, scale_h)

        out_width = int(target_width * scale)
        out_height = int(target_height * scale)

        dst_marker_pts_adjusted = (dst_marker_pts - np.array([min_x, min_y], dtype="float32")) * scale
        
        H, _ = cv2.findHomography(m_corners, dst_marker_pts_adjusted)
        warped = cv2.warpPerspective(image, H, (out_width, out_height))

        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        return warped, annotated, True

def main():
    parser = argparse.ArgumentParser(description="Whiteboard perspective correction using ArUco markers.")
    parser.add_argument("-i", "--input", help="Path to the input image (used for static mode)")
    parser.add_argument("-c", "--camera", type=int, default=-1, help="Webcam device index (e.g., 0). Overrides input image if provided.")
    parser.add_argument("-p", "--pause-on-fail", action="store_true", help="Pause the camera if correction fails.")
    parser.add_argument("-w", "--max-width", type=int, default=1920, help="Maximum width of the output image")
    parser.add_argument("-t", "--max-height", type=int, default=1080, help="Maximum height of the output image")
    parser.add_argument("-ao", "--aruco-order", action="store_true", help="Sort corners using ArUco IDs (Lowest ID = TL, 2nd = TR, 3rd = BR, Highest = BL)")
    args = parser.parse_args()

    if args.input is None and args.camera < 0:
        parser.error("You must provide either an input image (-i) or a camera index (-c).")

    # Configure the ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    if args.camera >= 0:
        # --- WEBCAM MODE ---
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            return
            
        last_good = None
        print("Starting webcam mode. Press 'ESC' or 'q' to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            warped, annotated, success = process_image(frame, detector, args)

            if success:
                last_good = warped.copy()
                display_img = warped
            else:
                if not args.pause_on_fail:
                    display_img = frame
                if last_good is not None:
                    display_img = last_good.copy()
                    # Draw a 10x10 red square so it is visible on standard resolutions
                    cv2.rectangle(display_img, (0, 0), (10, 10), (0, 0, 255), -1)
                else:
                    display_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_img, "Waiting for markers...", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Detected Markers (Webcam)", annotated)
            cv2.imshow("Corrected Whiteboard", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:  # ESC or q
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # --- STATIC IMAGE MODE ---
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not load image from {args.input}")
            return

        warped, annotated, success = process_image(image, detector, args)

        if not success:
            print("Error: Could not find exactly 1, 2, or 4 markers.")
            cv2.imshow("Found Markers", annotated)
            cv2.waitKey(0)
            return
            
        print(f"Output resolution: {warped.shape[1]}x{warped.shape[0]}")

        viewer1 = napari.Viewer(title="Detected Markers")
        viewer1.add_image(annotated, rgb=True)

        viewer2 = napari.Viewer(title="Corrected Whiteboard")
        viewer2.add_image(warped, rgb=True)

        print("Close the viewers to exit...")
        napari.run()

if __name__ == "__main__":
    main()