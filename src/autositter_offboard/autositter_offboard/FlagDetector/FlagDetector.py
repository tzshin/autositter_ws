import cv2 as cv
import numpy as np
import sys
import math
import easyocr
import logging


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# logging.disable(logging.CRITICAL) #disable while debug


class FlagDetector:
    def __init__(self, reference_path, hsv_ranges=None, safe_column_width_ratio=0.5):
        """
        Initialize the detector with the reference image path.
        """
        self.reference_contour = self.get_reference_contour(reference_path)
        self.best_contour = None  # Initialize best_contour
        self.corners = None       # Initialize corners
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.display_width = 1280   # Will be set dynamically
        self.display_height = 720  # Will be set dynamically
        self.point_LM = None
        self.safe_column_width_ratio = safe_column_width_ratio
        self.compensate_angle = None
        # Define default HSV ranges for red if none are provided
        if hsv_ranges is None:
            self.hsv_ranges = [
                {"lower": [0, 120, 70], "upper": [10, 255, 255]},
                {"lower": [170, 120, 70], "upper": [180, 255, 255]}
            ]
        else:
            self.hsv_ranges = hsv_ranges

    def draw_safe_column(self, frame):
        # Draw the safe column on the frame
        frame_center = (int(self.display_width / 2),
                        int(self.display_height / 2))
        radius_pixels = self.safe_column_width_ratio / 2 * self.display_height
        radius_pixels_int = int(radius_pixels)
        cv.circle(frame, frame_center, radius_pixels_int,
                  (0, 255, 255), 2)  # Yellow circle
        cv.circle(frame, frame_center, 5, (0, 255, 255), -1)  # Center point

    def get_reference_contour(self, reference_path):
        """
        Load the reference image and extract its contour.
        """
        reference = cv.imread(reference_path)
        if reference is None:
            logging.error("Error loading reference image.")
            sys.exit()

        # Convert to grayscale and apply Gaussian blur
        reference_gray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
        reference_blurred = cv.GaussianBlur(reference_gray, (5, 5), 0)
        reference_edges = cv.Canny(reference_blurred, 50, 150)
        reference_contours, _ = cv.findContours(
            reference_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Check if any contours are found
        if len(reference_contours) == 0:
            logging.error("No contours found in the reference image.")
            sys.exit()

        # Select the largest contour based on area
        largest_contour = max(reference_contours, key=cv.contourArea)
        return largest_contour

    def preprocess_frame(self, frame):
        """
        Preprocess the input frame to extract edges.
        """

        self.display_height, self.display_width = frame.shape[:2]

        # Extract regions based on HSV ranges
        frame_masked = self.get_masked_regions(frame)

        # cv.imshow('frame_masked', cv.resize(frame_masked,
        #           (self.display_width, self.display_height)))
        # Apply morphological operations
        frame_morpho = self.morphological(frame_masked)

        # Convert to grayscale and apply Gaussian blur
        gray = cv.cvtColor(frame_morpho, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (3, 3), 0)

        # Apply edge detection (Canny)
        edges = cv.Canny(blurred, 50, 150)
        # cv.imshow('frame', cv.resize(frame, (self.display_width, self.display_height)))
        return edges

    def get_masked_regions(self, image):
        """
        Extract regions from the image based on provided HSV ranges.

        Args:
            image (numpy.ndarray): The original BGR image.

        Returns:
            numpy.ndarray: Image with masked regions.
        """
        # Convert from BGR to HSV color space
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        mask = None
        for range_dict in self.hsv_ranges:
            lower = np.array(range_dict["lower"], dtype=np.uint8)
            upper = np.array(range_dict["upper"], dtype=np.uint8)
            current_mask = cv.inRange(hsv, lower, upper)
            if mask is None:
                mask = current_mask
            else:
                mask = cv.bitwise_or(mask, current_mask)

        # Extract the regions based on the combined mask
        result = cv.bitwise_and(image, image, mask=mask)

        return result

    def morphological(self, image):
        """
        Apply morphological operations to reduce noise.
        """
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        image = cv.dilate(image, kernel, iterations=1)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE,
                                kernel)  # Close small gaps
        image = cv.morphologyEx(image, cv.MORPH_CLOSE,
                                kernel)  # Close small gaps
        image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel=cv.getStructuringElement(
            cv.MORPH_ELLIPSE, (1, 1)))   # Remove small noise

        # cv.imshow('frame_masked', cv.resize(
        #     image, (self.display_width, self.display_height)))

        return image

    def compute_min_angle(self, polygon_points):
        """
        Compute the point with the minimum internal angle in a polygon.
        """
        min_angle = None
        min_angle_point = None
        num_points = len(polygon_points)
        for i in range(num_points):
            # Current, previous, and next points
            p_curr = np.array(polygon_points[i])
            p_prev = np.array(polygon_points[(i - 1) % num_points])
            p_next = np.array(polygon_points[(i + 1) % num_points])

            # Vectors from current point to previous and next points
            v1 = p_prev - p_curr
            v2 = p_next - p_curr

            # Handle zero-length vectors
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                continue

            # Normalize the vectors
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)

            # Compute the angle in radians
            dot_product = np.dot(v1_norm, v2_norm)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            # Update the minimum angle if necessary
            if (min_angle is None) or (angle < min_angle):
                min_angle = angle
                min_angle_point = i

        return min_angle_point

    def sort_corners(self, corners, center):
        """
        Sort the corners of the polygon based on their angle relative to the center.
        """
        # Convert corners and center to NumPy arrays
        corners_array = np.array(corners)
        center_array = np.array(center)
        # Subtract center coordinates
        centered = corners_array - center_array
        # Compute angles
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        # Sort by angles
        sorted_indices = angles.argsort()
        sorted_corners = corners_array[sorted_indices]
        return [tuple(point) for point in sorted_corners]

    def calculate_moments_center(self, frame):
        """
        Calculate and mark the center of the best contour on the frame.
        """
        M = cv.moments(self.best_contour)
        if M['m00'] == 0:
            M['m00'] = 1
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        self.center = (center_x, center_y)
        cv.circle(frame, self.center, 5, (255, 0, 0), -1)
        return frame

    def calculate_upper_vector(self, frame):
        """
        Calculate the orientation vectors of the best matching contour.
        """
        frame = self.calculate_moments_center(frame)

        # Extract corner points
        corners = [tuple(point[0]) for point in self.best_contour]

        # Sort the corners
        corners = self.sort_corners(corners, self.center)

        # Store the corners for use in extract_and_ocr
        self.corners = corners

        # Convert corners to NumPy arrays
        corners_array = [np.array(corner) for corner in corners]

        # Compute the minimum internal angle
        min_internal_angle_point = self.compute_min_angle(corners)

        if min_internal_angle_point is None:
            logging.warning("Could not compute minimum internal angle.")
            return frame

        # Find lower center (self.point_LM)
        point_LL = corners_array[(min_internal_angle_point + 2) % 5]
        point_LR = corners_array[(min_internal_angle_point + 3) % 5]
        self.point_LM = (point_LL + point_LR) / 2  # NumPy array

        # Compute vectors
        point_U = corners_array[min_internal_angle_point]
        self.vector_U = point_U - self.point_LM
        self.vector_R = point_LR - self.point_LM

        # Draw corner points and vectors
        for idx, corner in enumerate(corners):
            corner_int = tuple(map(int, corner))
            if idx == min_internal_angle_point:
                point_LM_int = tuple(map(int, self.point_LM))
                cv.circle(frame, corner_int, 5, (255, 0, 255), -1)
                cv.arrowedLine(frame, point_LM_int, corner_int, color=(
                    255, 100, 0), thickness=3, tipLength=0.05)
            else:
                cv.circle(frame, corner_int, 5, (255, 0, 0), -1)
            cv.putText(frame, f"P{idx+1}", corner_int,
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return frame

    def process_frame(self, frame, mode):
        """
        Process the frame to detect shapes and calculate orientation vectors.
        """
        self.is_flag = False
        self.error = None
        self.aiming = None
        self.detected_num = None
        self.is_within_safe_column = None  # Reset the flag for each frame
        self.best_contour = None
        self.pixels_per_meter = None

        frame_origin = frame.copy()
        # self.draw_safe_column(frame) #draw safe column every time frame input
        # Preprocess the frame
        edges = self.preprocess_frame(frame)

        # Find contours
        contours, _ = cv.findContours(
            edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if mode == 'bearing':
            frame = self.find_biggest_contour(frame, contours)
        elif mode == 'visual':
            frame = self.find_best_contour(frame, contours)
        else:
            logging.warning(f"Unknown mode '{mode}' selected.")
            return frame

        ################## - while detected flag ##################
        if self.best_contour is not None:
            self.is_flag = True

            if mode == 'bearing':
                self.corners = [tuple(point[0]) for point in self.best_contour]
                self.detected_num = self.direct_ocr(frame_origin, self.corners)
                # min_match_value = 1.0
                # if len(self.best_contour) == 5 and cv.matchShapes(self.best_contour, self.reference_contour, cv.CONTOURS_MATCH_I2, 0) < min_match_value:
                #     frame = self.process_flag(frame)
                #     self.detected_num = self.extract_and_ocr(frame_origin, self.corners)
            elif mode == 'visual':
                frame = self.process_flag(frame)
                self.detected_num = self.extract_and_ocr(
                    frame_origin, self.corners)

        # cv.imshow('frame', frame)
        return frame  # Return the processed frame for visualization

    def process_flag(self, frame):
        frame = self.calculate_upper_vector(frame)
        self.aiming = self.calculate_aiming()
        # Calculate error value
        error_x, error_y = self.calculate_distance_to_frame_center()
        if error_x is not None and error_y is not None:
            self.error = (float(error_x), float(error_y))
        else:
            self.error = (None, None)

        if self.pixels_per_meter is not None and self.display_height is not None:
            radius_pixels = self.safe_column_width_ratio / 2 * self.display_height
            radius_meters = radius_pixels / self.pixels_per_meter
            error_distance = math.sqrt(
                error_x**2 + error_y**2) if error_x and error_y else None
            # Set the boolean flag based on the error
            self.is_within_safe_column = error_distance < radius_meters if error_distance else False
        else:
            logging.error("Error: vector_R or display dimensions not set.")

    def find_biggest_contour(self, frame, contours):
        """
        Find the biggest contour and draw it on the frame.
        """
        min_area = 500  # Minimum area threshold
        if len(contours) == 0:
            logging.info("No contours found.")
            return frame
        else:
            # Find the largest contour by area
            biggest_contour = max(contours, key=cv.contourArea)
            if cv.contourArea(biggest_contour) > min_area:
                self.best_contour = biggest_contour
                self.calculate_moments_center(frame)
                # Get the perimeter of the contour
                perimeter = cv.arcLength(self.best_contour, True)
                # Approximate the contour shape
                self.best_contour = cv.approxPolyDP(
                    self.best_contour, 0.02 * perimeter, True)
                # Draw the largest contour on the frame
                cv.drawContours(frame, [self.best_contour], -1, (0, 255, 0), 3)
            else:
                logging.info(
                    "No matching contour found based biggest contour.")
            return frame

    def extract_roi_bounding_box(self, image, corners):
        """
        Extracts the ROI from the image by cropping based on the bounding rectangle of the contour.

        Args:
            image (numpy.ndarray): Original BGR image.
            corners (list or numpy.ndarray): List of corner points [(x1, y1), (x2, y2), ...].

        Returns:
            numpy.ndarray: Cropped image containing the ROI.
        """
        # Convert corners to a NumPy array
        corners = np.array(corners, dtype=np.int32)

        # Compute the bounding rectangle of the contour
        x, y, w, h = cv.boundingRect(corners)

        # Crop the image using the bounding rectangle
        roi_cropped = image[y:y+h, x:x+w]

        return roi_cropped

    def find_best_contour(self, frame, contours, expected_sides=5):
        """
        Find the contour that best matches the reference contour based on expected number of sides.
        """
        if len(contours) == 0:
            logging.info("No contours found.")
            return frame

        min_match_value = 1.0
        min_area = 500  # Minimum area threshold

        best_contour = None  # Initialize best_contour

        for contour in contours:
            # Get the perimeter of the contour
            perimeter = cv.arcLength(contour, True)

            # Approximate the contour shape
            approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
            # cv.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            # cv.imshow('frame', frame)
            # Check if the shape has the expected number of sides and sufficient area
            if cv.contourArea(contour) > min_area and len(approx) == expected_sides:
                # Compare shapes and get the match value
                match_value = cv.matchShapes(
                    approx, self.reference_contour, cv.CONTOURS_MATCH_I2, 0)

                # Update the minimum match value and store the corresponding contour
                if match_value < min_match_value:
                    min_match_value = match_value
                    best_contour = approx

        if best_contour is not None:
            cv.drawContours(frame, [best_contour], -1, (0, 255, 0), 3)
            self.best_contour = best_contour
            # cv.imshow('frame', frame)
        else:
            logging.info("No matching contour found based on Hu Moments.")

        return frame

    def extract_and_ocr(self, frame, corners):
        """
        Extract the ROI from the frame, correct its perspective, and perform OCR to detect numbers.
        """
        if not corners or len(corners) < 5:
            logging.warning("Insufficient corners for OCR extraction.")
            return None

        min_angle_idx = self.compute_min_angle(corners)
        if min_angle_idx is None:
            logging.warning(
                "Could not compute minimum internal angle for OCR extraction.")
            return None

        # Since points are ordered CW starting after min_angle_idx, define indices accordingly
        idx_right_upper = (min_angle_idx + 1) % 5
        idx_right_down = (min_angle_idx + 2) % 5
        idx_left_down = (min_angle_idx + 3) % 5
        idx_left_upper = (min_angle_idx + 4) % 5

        # Define source points using the known indices
        src_pts = np.array([
            corners[idx_right_upper],  # Right upper point
            corners[idx_right_down],   # Right down point
            corners[idx_left_down],    # Left down point
            corners[idx_left_upper]    # Left upper point
        ], dtype='float32')

        # Define the destination points to map to a rectangle
        width, height = 200, 120  # Increased size for better OCR accuracy
        dst_pts = np.array([
            [width - 1, 0],             # Top-right corner
            [width - 1, height - 1],    # Bottom-right corner
            [0, height - 1],            # Bottom-left corner
            [0, 0]                       # Top-left corner
        ], dtype='float32')

        # Compute the perspective transformation matrix
        try:
            M = cv.getPerspectiveTransform(src_pts, dst_pts)
        except cv.error as e:
            logging.error(f"Perspective transformation failed: {e}")
            return None

        # Apply the perspective transformation
        warped = cv.warpPerspective(frame, M, (width, height))

        # Preprocess the warped image for better OCR results
        # warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        # warped_thresh = cv.threshold(warped_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

        results = self.reader.readtext(
            warped, allowlist='0123456789', detail=1)
        # cv.imshow('warped', cv.resize(warped, (200, 120)))

        if not results:
            logging.info("No text detected in the warped image.")
            return None
        else:
            # Extract the number with the highest confidence score
            # x[2] is the confidence score
            best_result = max(results, key=lambda x: x[2])

            # Return the number with the highest confidence
            detected_text = best_result[1].strip()
            logging.info(
                f"OCR Detected Text: {detected_text} with confidence {best_result[2]:.2f}")
            return detected_text

    def rotate_image(self, image, angle):
        # Get the image dimensions (height, width)
        (h, w) = image.shape[:2]

        # Calculate the center of the image
        image_center = (w / 2, h / 2)

        # Get the rotation matrix for the given angle
        rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

        # Compute the sine and cosine of the rotation angle
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])

        # Compute the new width and height bounds
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to take into account the translation
        rot_mat[0, 2] += (new_w / 2) - image_center[0]
        rot_mat[1, 2] += (new_h / 2) - image_center[1]

        # Perform the rotation with the new bounds
        rotated_image = cv.warpAffine(
            image, rot_mat, (new_w, new_h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

        return rotated_image

    def direct_ocr(self, frame, corners):

        # Apply the perspective transformation
        cropped = self.extract_roi_bounding_box(frame, corners)
        # Preprocess the warped image for better OCR results
        cropped_gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)
        cropped_thresh = cv.threshold(
            cropped_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        cropped_rotate = self.rotate_image(
            cropped_thresh, self.compensate_angle)
        cropped_rotate = cv.resize(cropped_rotate, (200, 120))

        # cv.imshow('cropped_rotate', cropped_rotate)
        # cv.imshow('cropped', cropped)
        results = self.reader.readtext(
            cropped_rotate, allowlist='0123456789', detail=1)
        if not results:
            logging.info("No text detected in the warped image.")
            return None
        else:
            # Extract the number with the highest confidence score
            # x[2] is the confidence score
            best_result = max(results, key=lambda x: x[2])

            # Return the number with the highest confidence
            detected_text = best_result[1].strip()
            logging.info(
                f"OCR Detected Text: {detected_text} with confidence {best_result[2]:.2f}")
            return detected_text

    def calculate_distance_to_frame_center(self):
        """
        Calculate the distance from self.center to the frame center in meters.

        Returns:
            tuple: (error_x, error_y) in meters. Returns (None, None) if calculation cannot be performed.
        """
        if self.vector_R is None:
            logging.error("Error: vector_R is not set.")
            return (None, None)
        if self.center is None:
            logging.error("Error: center is not set.")
            return (None, None)

        # Compute pixel length of vector_R
        length_pixels_R = np.linalg.norm(self.vector_R)
        if length_pixels_R == 0:
            logging.error("Error: vector_R has zero length.")
            return (None, None)

        # Calculate pixels per meter (Scale Factor)
        self.pixels_per_meter = length_pixels_R / \
            0.5  # vector_R corresponds to 0.5 meters

        # Define frame center
        frame_center = (self.display_width / 2, self.display_height / 2)

        # Calculate pixel distance between self.center and frame_center
        delta_pixels = np.array(frame_center) - np.array(self.center)
        distance_pixels_x = delta_pixels[0]
        distance_pixels_y = delta_pixels[1]

        # Convert pixel distances to meters
        distance_meters_x = distance_pixels_x / self.pixels_per_meter
        distance_meters_y = distance_pixels_y / self.pixels_per_meter

        return (distance_meters_x, distance_meters_y)

    def calculate_aiming(self):
        """
        Calculate the angle between the positive X-axis and vector_U in radians.

        Returns:
            float: Angle in radians if vector_U is defined.
            None: If vector_U is not set or has zero length.
        """
        if self.vector_U is None:
            logging.error("Error: vector_U is not set.")
            return None

        # Ensure vector_U has non-zero length
        norm = np.linalg.norm(self.vector_U)
        if norm == 0:
            logging.error("Error: vector_U has zero length.")
            return None

        # Calculate the angle using arctan2
        angle_rad = math.atan2(
            self.vector_U[1], self.vector_U[0])  # Y, X order

        return angle_rad
