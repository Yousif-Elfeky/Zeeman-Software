import cv2
import numpy as np
from typing import Optional, Tuple

class ImageProcessor:
    def __init__(self):
        self.image = None
        self.processed_image = None
    
    def enhance_image(self):
        if self.image is None:
            raise ValueError("No image loaded.")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(blurred)
        
        return self.processed_image
    
    def analyze_ring_boundaries(self, processed_image: np.ndarray, center_x: int, center_y: int, r_center_estimate: float,
                                radial_search_width: int = 20, num_angles: int = 360) -> Optional[Tuple[float, float]]:
        if processed_image is None or processed_image.ndim != 2:
            return None

        height, width = processed_image.shape
        
        half_width = radial_search_width / 2.0
        r_scan_start = max(0.0, r_center_estimate - half_width)
        r_scan_end = min(float(min(height, width) / 2.0 - 1.0), r_center_estimate + half_width)

        if r_scan_start >= r_scan_end: 
            return None 
            
        sampled_radii = np.arange(np.floor(r_scan_start), np.ceil(r_scan_end))
        if len(sampled_radii) == 0:
            return None

        radial_profile = np.zeros(len(sampled_radii))
        
        for i_angle in range(num_angles):
            angle = 2 * np.pi * i_angle / num_angles
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            for i, r_val in enumerate(sampled_radii):
                x = int(round(center_x + r_val * cos_a))
                y = int(round(center_y + r_val * sin_a))
                
                if 0 <= x < width and 0 <= y < height:
                    radial_profile[i] += processed_image[y, x]
        
        if num_angles > 0:
            radial_profile /= num_angles
        else:
            return None

        if not radial_profile.any():
            return None

        peak_idx_local = np.argmax(radial_profile)
        
        profile_min_val = np.min(radial_profile)
        peak_value = radial_profile[peak_idx_local]
        
        if peak_value <= profile_min_val:
            return None

        threshold = profile_min_val + (peak_value - profile_min_val) * 0.5 

        r_inner_local_idx = peak_idx_local
        while r_inner_local_idx > 0 and radial_profile[r_inner_local_idx - 1] > threshold:
            r_inner_local_idx -= 1
        
        r_outer_local_idx = peak_idx_local
        while r_outer_local_idx < len(radial_profile) - 1 and radial_profile[r_outer_local_idx + 1] > threshold:
            r_outer_local_idx += 1
            
        r_inner_abs = sampled_radii[r_inner_local_idx]
        r_outer_abs = sampled_radii[r_outer_local_idx]

        if r_inner_abs > r_outer_abs:
            return None 

        return float(r_inner_abs), float(r_outer_abs)

    def auto_detect_radius_refined(self, processed_image: np.ndarray, initial_center_x: int, initial_center_y: int, radius_lower_limit: int, radius_upper_limit: int, center_search_window_half_size: int = 5) -> Optional[dict]:
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
        
        weighted_circles = []
        
        for center_x, center_y in candidate_centers:
            mask = np.zeros(processed_image.shape, dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius_upper_limit, 255, -1)
            cv2.circle(mask, (center_x, center_y), radius_lower_limit, 0, -1)
            
            roi_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
            
            circles = cv2.HoughCircles(
                roi_image,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(1, int(radius_lower_limit / 2)), 
                param1=100,
                param2=10, 
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
                    cv2.circle(circle_mask, (int(round(x)), int(round(y))), int(round(r)), 255, 2) # Thickness changed to 2
                    
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
                        0.2 * distance_weight +  
                        0.5 * edge_weight +      
                        0.2 * completeness_weight +
                        0.1 * center_weight      
                    )
                    
                    weighted_circles.append((final_weight, int(round(x)), int(round(y)), int(round(r))))
        
        if weighted_circles:
            sorted_circles = sorted(weighted_circles, key=lambda item: item[0], reverse=True)
            best_circle_weight, best_circle_x, best_circle_y, best_circle_r = sorted_circles[0]
 
            final_center_x = best_circle_x
            final_center_y = best_circle_y
            
            ring_boundaries = self.analyze_ring_boundaries(
                processed_image, 
                int(round(best_circle_x)),
                int(round(best_circle_y)),
                best_circle_r
            )

            r_inner, r_outer = None, None
            if ring_boundaries:
                r_inner, r_outer = ring_boundaries

            return {
                'center_x': final_center_x,
                'center_y': final_center_y,
                'radius_centerline': best_circle_r, # Radius from Houg
                'radius_inner': r_inner,          
                'radius_outer': r_outer,          
                'weight': best_circle_weight
            }
         
        return None
