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
        """Detect and measure the spectral lines, returning the center (x,y) and radius (r) of the best circle found.
        
        This method evaluates multiple potential center points around the manually specified one,
        assigns weights to each based on circle quality, and selects the one with the highest weight.
        """
        if not (0 <= radius_lower_limit < radius_upper_limit):
            raise ValueError("Radius limits are invalid.")
        if processed_image is None:
            raise ValueError("Processed image is not available.")
        if processed_image.ndim != 2:
            raise ValueError("Processed image must be grayscale.")
        if center_search_window_half_size < 0:
            raise ValueError("Center search window half size must be non-negative.")

        # Define a larger search window for candidate center points
        search_grid_size = center_search_window_half_size * 2
        
        # Create a list of candidate center points to evaluate
        candidate_centers = []
        for dx in range(-search_grid_size, search_grid_size + 1, 2):  # Step by 2 to reduce computation
            for dy in range(-search_grid_size, search_grid_size + 1, 2):  # Step by 2 to reduce computation
                candidate_centers.append((initial_center_x + dx, initial_center_y + dy))
        
        # Add the initial center point to ensure it's evaluated
        if (initial_center_x, initial_center_y) not in candidate_centers:
            candidate_centers.append((initial_center_x, initial_center_y))
        
        # Dictionary to store weighted results for each candidate center
        weighted_results = {}
        
        # Evaluate each candidate center point
        for center_x, center_y in candidate_centers:
            # Create an annular mask centered at the current candidate center
            mask = np.zeros(processed_image.shape, dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius_upper_limit, 255, -1)
            cv2.circle(mask, (center_x, center_y), radius_lower_limit, 0, -1)
            
            roi_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
            
            # Adjust HoughCircles parameters
            circles = cv2.HoughCircles(
                roi_image,
                cv2.HOUGH_GRADIENT,
                dp=1,  # Standard resolution
                minDist=max(1, int(radius_lower_limit / 4)),
                param1=100,  # Canny edge upper threshold (standard value)
                param2=15,   # Accumulator threshold (lowered)
                minRadius=radius_lower_limit,
                maxRadius=radius_upper_limit
            )
            
            if circles is not None:
                # Process detected circles for this candidate center
                detected_circles = circles[0, :]
                
                for circle in detected_circles:
                    x, y, r = circle[0], circle[1], circle[2]
                    
                    # Calculate weights based on multiple factors
                    # 1. Distance from initial center (closer is better)
                    distance_from_initial = np.sqrt((x - initial_center_x)**2 + (y - initial_center_y)**2)
                    distance_weight = 1.0 / (1.0 + 0.1 * distance_from_initial)  # Inverse distance weight
                    
                    # 2. Circle quality based on edge strength
                    # Create a circle mask to extract the circle edge
                    circle_mask = np.zeros(processed_image.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (int(round(x)), int(round(y))), int(round(r)), 255, 1)
                    
                    # Extract edge pixels and calculate their mean intensity
                    edge_pixels = cv2.bitwise_and(processed_image, processed_image, mask=circle_mask)
                    non_zero_pixels = edge_pixels[edge_pixels > 0]
                    
                    if len(non_zero_pixels) > 0:
                        edge_strength = np.mean(non_zero_pixels)
                        edge_weight = edge_strength / 255.0  # Normalize to 0-1 range
                    else:
                        edge_weight = 0.0
                    
                    # 3. Circle completeness (how many edge pixels are detected)
                    circle_perimeter = 2 * np.pi * r
                    completeness_weight = min(1.0, len(non_zero_pixels) / max(1, circle_perimeter))
                    
                    # 4. Proximity to candidate center
                    center_proximity = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    center_weight = 1.0 / (1.0 + center_proximity)
                    
                    # Calculate final weight as a combination of all factors
                    # Adjust these coefficients to change the importance of each factor
                    final_weight = (
                        0.3 * distance_weight +
                        0.4 * edge_weight +
                        0.2 * completeness_weight +
                        0.1 * center_weight
                    )
                    
                    # Store the result with its weight
                    circle_key = (int(round(x)), int(round(y)), int(round(r)))
                    weighted_results[circle_key] = final_weight
        
        # Select the circle with the highest weight
        if weighted_results:
            best_circle = max(weighted_results.items(), key=lambda item: item[1])[0]
            return best_circle
        
        return None
