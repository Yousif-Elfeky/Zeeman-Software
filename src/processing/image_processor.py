import cv2
import numpy as np
from typing import Optional, Tuple

class ImageProcessor:
    def __init__(self):
        self.image = None
        self.processed_image = None
    
    def load_image(self, file_path):
        self.image = cv2.imread(file_path)
        if self.image is None:
            raise ValueError("Failed to load image")
        return self.image
    
    def enhance_image(self):
        if self.image is None:
            raise ValueError("No image loaded")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(blurred)
        
        return self.processed_image
    
    def detect_spectral_lines(self, processed_image: np.ndarray, initial_center_x: int, initial_center_y: int, radius_lower_limit: int, radius_upper_limit: int, center_search_window_half_size: int = 5) -> Optional[Tuple[int, int, int]]:
        if not (0 <= radius_lower_limit < radius_upper_limit):
            raise ValueError("Radius limits are invalid.")
        if processed_image is None:
            raise ValueError("Processed image is not available.")
        if processed_image.ndim != 2:
            raise ValueError("Processed image must be grayscale.")
        if center_search_window_half_size < 0:
            raise ValueError("Center search window half size must be non-negative.")

        search_grid_size = center_search_window_half_size * 2
        
        candidate_centers = []
        for dx in range(-search_grid_size, search_grid_size + 1, 2):  
            for dy in range(-search_grid_size, search_grid_size + 1, 2):
                candidate_centers.append((initial_center_x + dx, initial_center_y + dy))
        
        if (initial_center_x, initial_center_y) not in candidate_centers:
            candidate_centers.append((initial_center_x, initial_center_y))
        
        weighted_results = {}
        
        for center_x, center_y in candidate_centers:
            mask = np.zeros(processed_image.shape, dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius_upper_limit, 255, -1)
            cv2.circle(mask, (center_x, center_y), radius_lower_limit, 0, -1)
            
            roi_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
            
            circles = cv2.HoughCircles(
                roi_image,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(1, int(radius_lower_limit / 4)),
                param1=100,
                param2=15,
                minRadius=radius_lower_limit,
                maxRadius=radius_upper_limit
            )
            
            if circles is not None:
                detected_circles = circles[0, :]
                
                for circle in detected_circles:
                    x, y, r = circle[0], circle[1], circle[2]
                    
                    distance_from_initial = np.sqrt((x - initial_center_x)**2 + (y - initial_center_y)**2)
                    distance_weight = 1.0 / (1.0 + 0.1 * distance_from_initial)  # Inverse distance weight
                    
                    circle_mask = np.zeros(processed_image.shape, dtype=np.uint8)
                    cv2.circle(circle_mask, (int(round(x)), int(round(y))), int(round(r)), 255, 1)
                    
                    edge_pixels = cv2.bitwise_and(processed_image, processed_image, mask=circle_mask)
                    non_zero_pixels = edge_pixels[edge_pixels > 0]
                    
                    if len(non_zero_pixels) > 0:
                        edge_strength = np.mean(non_zero_pixels)
                        edge_weight = edge_strength / 255.0  
                    else:
                        edge_weight = 0.0
                    
                    circle_perimeter = 2 * np.pi * r
                    completeness_weight = min(1.0, len(non_zero_pixels) / max(1, circle_perimeter))
                    
                    center_proximity = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    center_weight = 1.0 / (1.0 + center_proximity)
                    
                    final_weight = (
                        0.3 * distance_weight +
                        0.4 * edge_weight +
                        0.2 * completeness_weight +
                        0.1 * center_weight
                    )
                    
                    circle_key = (int(round(x)), int(round(y)), int(round(r)))
                    weighted_results[circle_key] = final_weight
        
        if weighted_results:
            best_circle = max(weighted_results.items(), key=lambda item: item[1])[0]
            return best_circle
        
        return None
