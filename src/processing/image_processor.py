import cv2
import numpy as np
from typing import Optional

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
    
    def detect_spectral_lines(self, processed_image: np.ndarray, center_x: int, center_y: int, radius_lower_limit: int, radius_upper_limit: int) -> Optional[int]:
        """Detect and measure the radii of spectral lines using Hough Circle Transform in an annular region."""
        if not (0 <= radius_lower_limit < radius_upper_limit):
            raise ValueError("Radius limits are invalid.")

        if processed_image is None:
            raise ValueError("Processed image is not available.")
        
        if processed_image.ndim != 2:
            # Assuming processed_image should be grayscale
            raise ValueError("Processed image must be grayscale.")

        # Create an annular mask
        mask = np.zeros(processed_image.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius_upper_limit, 255, -1)
        cv2.circle(mask, (center_x, center_y), radius_lower_limit, 0, -1)
        
        # Apply the mask to the image
        roi_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
        
        # Detect circles using Hough Circle Transform
        # Parameters might need tuning:
        # dp: Inverse ratio of the accumulator resolution to the image resolution.
        # minDist: Minimum distance between the centers of the detected circles.
        # param1: Higher threshold for the Canny edge detector.
        # param2: Accumulator threshold for the circle centers at the detection stage.
        # minRadius: Minimum circle radius.
        # maxRadius: Maximum circle radius.
        circles = cv2.HoughCircles(
            roi_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=radius_upper_limit * 2, # Ensure circles are reasonably separated
            param1=100, # Canny edge upper threshold
            param2=20,  # Accumulator threshold
            minRadius=radius_lower_limit,
            maxRadius=radius_upper_limit
        )
        
        if circles is not None:
            # Circles are returned as a float32 array: [[x, y, radius], ...]
            # Convert to integers and return the radius of the first detected circle
            circles = np.uint16(np.around(circles))
            # For simplicity, return the radius of the first circle found.
            # Additional logic could be added here to select the "best" circle
            # (e.g., based on proximity to expected center or other criteria).
            return int(circles[0, 0, 2]) 
            
        return None
