import cv2
import numpy as np
from typing import Optional, Tuple

class ImageProcessor:
    def __init__(self):
        self.image = None
        self.processed_image = None
    
    def load_image(self, file_path):
        """Load and preprocess the image."""
        self.image = cv2.imread(file_path)
        if self.image is None:
            raise ValueError("Failed to load image")
        return self.image
    
    def enhance_image(self):
        """Enhance image quality through noise reduction and contrast adjustment."""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(blurred)
        
        return self.processed_image
    
    def detect_spectral_lines(self, processed_image: np.ndarray, initial_center_x: int, initial_center_y: int, radius_lower_limit: int, radius_upper_limit: int, center_search_window_half_size: int = 5) -> Optional[Tuple[int, int, int]]:
        """Detect and measure the spectral lines, returning the center (x,y) and radius (r) of the best circle found."""
        if not (0 <= radius_lower_limit < radius_upper_limit):
            raise ValueError("Radius limits are invalid.")
        if processed_image is None:
            raise ValueError("Processed image is not available.")
        if processed_image.ndim != 2:
            raise ValueError("Processed image must be grayscale.")
        if center_search_window_half_size < 0:
            raise ValueError("Center search window half size must be non-negative.")

        # Create an annular mask centered at the initial_center_x, initial_center_y
        # The HoughCircles will search within this masked region.
        # The actual filtering for center_search_window_half_size happens *after* circle detection.
        mask = np.zeros(processed_image.shape, dtype=np.uint8)
        cv2.circle(mask, (initial_center_x, initial_center_y), radius_upper_limit, 255, -1)
        cv2.circle(mask, (initial_center_x, initial_center_y), radius_lower_limit, 0, -1)
        
        roi_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
        
        # Adjust HoughCircles parameters
        # minDist: max(1, int(radius_lower_limit / 4)) to allow detecting circles that might be close if center shifts.
        # param2: 15 (lowered accumulator threshold to be more lenient)
        circles = cv2.HoughCircles(
            roi_image,
            cv2.HOUGH_GRADIENT,
            dp=1, # Standard resolution
            minDist=max(1, int(radius_lower_limit / 4)), 
            param1=100, # Canny edge upper threshold (standard value)
            param2=15,  # Accumulator threshold (lowered)
            minRadius=radius_lower_limit,
            maxRadius=radius_upper_limit
        )
        
        if circles is not None:
            # circles is [[[x, y, r], [x, y, r], ...]]
            detected_circles = circles[0, :] 
            
            valid_circles = []
            for c in detected_circles:
                x, y, r = c[0], c[1], c[2]
                # Filter by center search window
                if (abs(x - initial_center_x) <= center_search_window_half_size and
                    abs(y - initial_center_y) <= center_search_window_half_size):
                    valid_circles.append((x, y, r))
            
            if not valid_circles:
                return None

            # Select the best circle: closest to initial_center_x, initial_center_y
            # If multiple are equally close, this will pick the first one encountered by min().
            # A secondary sort key (e.g., by radius size) could be added if needed.
            best_circle = min(valid_circles, key=lambda c_val: np.sqrt((c_val[0] - initial_center_x)**2 + (c_val[1] - initial_center_y)**2))
            
            return int(round(best_circle[0])), int(round(best_circle[1])), int(round(best_circle[2]))
            
        return None
