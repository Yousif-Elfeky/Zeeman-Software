import cv2
import numpy as np
from scipy import optimize

class ZeemanRingDetector:
    def __init__(self):
        self.image = None
        self.processed_image = None
        self.hue_filtered = None
        self.edge_detected = None
        
    def load_image(self, image):
        self.image = image.copy()
        return self.image
    
    def preprocess_image(self):
        if self.image is None:
            raise ValueError("No image loaded")
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        
        lower_magenta = np.array([140, 30, 50])
        upper_magenta = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_magenta, upper_magenta)
        
        lower_pink = np.array([0, 30, 50])
        upper_pink = np.array([20, 255, 255])
        mask2 = cv2.inRange(hsv, lower_pink, upper_pink)
        
        color_mask = cv2.bitwise_or(mask1, mask2)
        
        enhanced = self.image.copy()
        enhanced[color_mask == 0] = [0, 0, 0]
        
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        equalized = cv2.equalizeHist(gray)
        
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        self.hue_filtered = color_mask
        self.edge_detected = thresh
        
        self.processed_image = processed
        
        return self.processed_image
    
    def _circle_intensity(self, image, center_x, center_y, radius, width=2):
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = (dist_from_center >= (radius - width/2)) & (dist_from_center <= (radius + width/2))
        
        if center_x + radius >= w or center_x - radius < 0 or center_y + radius >= h or center_y - radius < 0:
            outside_penalty = 0.5
        else:
            outside_penalty = 1.0
        
        if np.sum(mask) > 0:
            mean_intensity = np.mean(image[mask]) * outside_penalty
            
            points = np.argwhere(mask & (image > 0))
            if len(points) > 10:
                distances = np.sqrt(np.sum((points - [center_y, center_x])**2, axis=1))
                circularity = 1.0 / (1.0 + np.std(distances) / radius)
                
                return mean_intensity * circularity
            
            return mean_intensity
        return 0
    
    def _find_optimal_radius(self, center_x, center_y, min_radius, max_radius, step=1):
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        min_radius = max(1, min_radius)
        max_radius = max(min_radius + 5, max_radius)
        
        radii = np.arange(min_radius, max_radius, step)
        intensities = [self._circle_intensity(self.processed_image, center_x, center_y, r) for r in radii]
        
        peaks = []
        for i in range(1, len(intensities)-1):
            if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1]:
                peaks.append((radii[i], intensities[i]))
        
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        if peaks:
            return peaks[0][0]
        
        max_idx = np.argmax(intensities)
        return radii[max_idx]
    
    def _refine_center(self, center_x, center_y, search_radius=20):
        if self.processed_image is None:
            self.preprocess_image()
            
        h, w = self.processed_image.shape[:2]
        best_x, best_y = center_x, center_y
        best_score = 0
        
        x_min = max(0, int(center_x - search_radius))
        x_max = min(w, int(center_x + search_radius))
        y_min = max(0, int(center_y - search_radius))
        y_max = min(h, int(center_y + search_radius))
        
        for x in range(x_min, x_max, 2):
            for y in range(y_min, y_max, 2):
                score = self._measure_symmetry(x, y)
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y
        
        return best_x, best_y
    
    def _measure_symmetry(self, x, y, num_angles=8, radius=100):
        if self.processed_image is None:
            return 0
            
        h, w = self.processed_image.shape[:2]
        profiles = []
        
        for i in range(num_angles):
            angle = i * (2 * np.pi / num_angles)
            profile = []
            
            for r in range(1, radius, 2):
                px = int(x + r * np.cos(angle))
                py = int(y + r * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    profile.append(self.processed_image[py, px])
                else:
                    break
            
            profiles.append(profile)
        
        symmetry_score = 0
        for i in range(num_angles):
            opposite_idx = (i + num_angles//2) % num_angles
            p1 = profiles[i]
            p2 = profiles[opposite_idx]
            
            min_len = min(len(p1), len(p2))
            if min_len > 0:
                p1_arr = np.array(p1[:min_len])
                p2_arr = np.array(p2[:min_len][::-1])
                
                if np.std(p1_arr) > 0 and np.std(p2_arr) > 0:
                    corr = np.corrcoef(p1_arr, p2_arr)[0, 1]
                    symmetry_score += max(0, corr)
        
        return symmetry_score
    
    def detect_rings(self, center_x, center_y, inner_limit=None, middle_limit=None, outer_limit=None):
        if self.processed_image is None:
            self.preprocess_image()
        
        h, w = self.processed_image.shape[:2]
        
        search_radius = min(h, w) // 2
        
        x1 = max(0, int(center_x - search_radius))
        y1 = max(0, int(center_y - search_radius))
        x2 = min(w, int(center_x + search_radius))
        y2 = min(h, int(center_y + search_radius))
        
        roi = self.processed_image[y1:y2, x1:x2]
        
        min_radius = 10
        max_radius = search_radius - 10
        
        if inner_limit is not None:
            min_inner = max(min_radius, inner_limit * 0.7)
        else:
            min_inner = min_radius
            
        if outer_limit is not None:
            max_outer = min(max_radius, outer_limit * 1.3)
        else:
            max_outer = max_radius
            
        manual_inner = inner_limit
        manual_middle = middle_limit
        manual_outer = outer_limit
        
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(20, int(min_inner * 0.5)),
            param1=80,
            param2=20,
            minRadius=int(min_inner),
            maxRadius=int(max_outer)
        )
        
        all_circles = []
        
        circles1 = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(20, int(min_inner * 0.5)),
            param1=80,
            param2=20,
            minRadius=int(min_inner),
            maxRadius=int(max_outer)
        )
        
        circles2 = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(10, int(min_inner * 0.3)),
            param1=50,
            param2=15,
            minRadius=int(min_inner),
            maxRadius=int(max_outer)
        )
        
        circles3 = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=0.5,
            minDist=max(10, int(min_inner * 0.3)),
            param1=70,
            param2=25,
            minRadius=int(min_inner),
            maxRadius=int(max_outer)
        )
        
        circles4 = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=5,
            param1=30,
            param2=10,
            minRadius=int(min_inner),
            maxRadius=int(max_outer)
        )
        
        mid_radius = (min_inner + max_outer) / 2
        circles5 = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(10, int(mid_radius * 0.2)),
            param1=60,
            param2=15,
            minRadius=int(min_inner),
            maxRadius=int(max_outer)
        )
        
        try:
            circles6 = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT_ALT,
                dp=1.5,
                minDist=max(10, int(min_inner * 0.3)),
                param1=300,
                param2=0.9,
                minRadius=int(min_inner),
                maxRadius=int(max_outer)
            )
        except:
            circles6 = None
        
        if circles1 is not None:
            all_circles.extend(circles1[0])
        if circles2 is not None:
            all_circles.extend(circles2[0])
        if circles3 is not None:
            all_circles.extend(circles3[0])
        if circles4 is not None:
            all_circles.extend(circles4[0])
        if circles5 is not None:
            all_circles.extend(circles5[0])
        if circles6 is not None:
            all_circles.extend(circles6[0])
        
        if not all_circles:
            if manual_inner is not None or manual_middle is not None or manual_outer is not None:
                inner_r = manual_inner if manual_inner is not None else (manual_middle * 0.7 if manual_middle is not None else (manual_outer * 0.5 if manual_outer is not None else search_radius * 0.3))
                middle_r = manual_middle if manual_middle is not None else ((inner_r + (manual_outer if manual_outer is not None else inner_r * 2)) / 2)
                outer_r = manual_outer if manual_outer is not None else (middle_r * 1.4)
                
                return {
                    'inner': inner_r,
                    'middle': middle_r,
                    'outer': outer_r,
                    'center_x': center_x,
                    'center_y': center_y
                }
            else:
                return {
                    'inner': search_radius * 0.3,
                    'middle': search_radius * 0.5,
                    'outer': search_radius * 0.7,
                    'center_x': center_x,
                    'center_y': center_y
                }
        
        detected_circles = []
        for circle in all_circles:
            adjusted_x = circle[0] + x1
            adjusted_y = circle[1] + y1
            radius = circle[2]
            
            dist_from_center = np.sqrt((adjusted_x - center_x)**2 + (adjusted_y - center_y)**2)
            
            if dist_from_center < search_radius * 0.3:
                detected_circles.append((adjusted_x, adjusted_y, radius))
        
        if not detected_circles:
            if manual_inner is not None or manual_middle is not None or manual_outer is not None:
                inner_r = manual_inner if manual_inner is not None else (manual_middle * 0.7 if manual_middle is not None else (manual_outer * 0.5 if manual_outer is not None else search_radius * 0.3))
                middle_r = manual_middle if manual_middle is not None else ((inner_r + (manual_outer if manual_outer is not None else inner_r * 2)) / 2)
                outer_r = manual_outer if manual_outer is not None else (middle_r * 1.4)
                
                return {
                    'inner': inner_r,
                    'middle': middle_r,
                    'outer': outer_r,
                    'center_x': center_x,
                    'center_y': center_y
                }
            else:
                return {
                    'inner': search_radius * 0.3,
                    'middle': search_radius * 0.5,
                    'outer': search_radius * 0.7,
                    'center_x': center_x,
                    'center_y': center_y
                }
        
        unique_circles = []
        radii = []
        for circle in sorted(detected_circles, key=lambda x: x[2]):
            is_unique = True
            for r in radii:
                if abs(circle[2] - r) / max(r, 1) < 0.1:
                    is_unique = False
                    break
            if is_unique:
                unique_circles.append(circle)
                radii.append(circle[2])
        
        if manual_inner is not None or manual_middle is not None or manual_outer is not None:
            def calc_adjustment_range(radius):
                return max(5, radius * 0.15) if radius is not None else 0
            
            inner_candidates = []
            middle_candidates = []
            outer_candidates = []
            
            if manual_inner is not None:
                inner_range = calc_adjustment_range(manual_inner)
                inner_candidates = [c for c in unique_circles if abs(c[2] - manual_inner) <= inner_range]
                # If no candidates within range, take the closest one
                if not inner_candidates and unique_circles:
                    inner_candidates = [min(unique_circles, key=lambda c: abs(c[2] - manual_inner))]
            
            if manual_middle is not None:
                middle_range = calc_adjustment_range(manual_middle)
                middle_candidates = [c for c in unique_circles if abs(c[2] - manual_middle) <= middle_range]
                # If no candidates within range, take the closest one
                if not middle_candidates and unique_circles:
                    middle_candidates = [min(unique_circles, key=lambda c: abs(c[2] - manual_middle))]
            
            if manual_outer is not None:
                outer_range = calc_adjustment_range(manual_outer)
                outer_candidates = [c for c in unique_circles if abs(c[2] - manual_outer) <= outer_range]
                # If no candidates within range, take the closest one
                if not outer_candidates and unique_circles:
                    outer_candidates = [min(unique_circles, key=lambda c: abs(c[2] - manual_outer))]
            
            # Select the best candidates from each group
            selected_circles = []
            
            # For inner radius, prefer smaller circles within range
            if inner_candidates:
                selected_circles.append(min(inner_candidates, key=lambda c: c[2]))
            
            # For middle radius, prefer circles in the middle of the range
            if middle_candidates:
                # Find the circle closest to the manual middle value
                selected_circles.append(min(middle_candidates, key=lambda c: abs(c[2] - manual_middle)))
            
            # For outer radius, prefer larger circles within range
            if outer_candidates:
                selected_circles.append(max(outer_candidates, key=lambda c: c[2]))
            
            # Add remaining circles if we don't have enough yet
            remaining = [c for c in unique_circles if c not in selected_circles]
            remaining.sort(key=lambda x: x[2])
            
            while len(selected_circles) < min(3, len(unique_circles)):
                if not remaining:
                    break
                selected_circles.append(remaining.pop(0))
            
            # Sort by radius
            selected_circles.sort(key=lambda x: x[2])
            
            # Extract radii and center coordinates
            radii = [circle[2] for circle in selected_circles]
            centers_x = [circle[0] for circle in selected_circles]
            centers_y = [circle[1] for circle in selected_circles]
            
            # Calculate center with limited movement from original center
            # Only allow small adjustments to the center point (max 5% of search radius)
            if centers_x and centers_y:
                raw_avg_x = sum(centers_x) / len(centers_x)
                raw_avg_y = sum(centers_y) / len(centers_y)
                
                # Limit how far the center can move from the manual center
                max_center_shift = search_radius * 0.05  # 5% of search radius
                
                # Calculate distance from original center
                center_shift = np.sqrt((raw_avg_x - center_x)**2 + (raw_avg_y - center_y)**2)
                
                # If shift is too large, scale it down
                if center_shift > max_center_shift:
                    scale_factor = max_center_shift / center_shift
                    # Move in the same direction but limit the distance
                    avg_x = center_x + (raw_avg_x - center_x) * scale_factor
                    avg_y = center_y + (raw_avg_y - center_y) * scale_factor
                else:
                    avg_x = raw_avg_x
                    avg_y = raw_avg_y
            else:
                # If no circles detected, use original center
                avg_x = center_x
                avg_y = center_y
        else:
            unique_circles.sort(key=lambda x: x[2])
            
            radii = [circle[2] for circle in unique_circles]
            centers_x = [circle[0] for circle in unique_circles]
            centers_y = [circle[1] for circle in unique_circles]
            
            if centers_x and centers_y:
                raw_avg_x = sum(centers_x) / len(centers_x)
                raw_avg_y = sum(centers_y) / len(centers_y)
                
                max_center_shift = search_radius * 0.05
                
                center_shift = np.sqrt((raw_avg_x - center_x)**2 + (raw_avg_y - center_y)**2)
                
                if center_shift > max_center_shift:
                    scale_factor = max_center_shift / center_shift
                    avg_x = center_x + (raw_avg_x - center_x) * scale_factor
                    avg_y = center_y + (raw_avg_y - center_y) * scale_factor
                else:
                    avg_x = raw_avg_x
                    avg_y = raw_avg_y
            else:
                avg_x = center_x
                avg_y = center_y
        
        if len(radii) >= 3:
            inner_radius = radii[0]
            middle_radius = radii[len(radii) // 2]
            outer_radius = radii[-1]
        elif len(radii) == 2:
            inner_radius = radii[0]
            outer_radius = radii[1]
            middle_radius = (inner_radius + outer_radius) / 2
        elif len(radii) == 1:
            middle_radius = radii[0]
            inner_radius = middle_radius * 0.7
            outer_radius = middle_radius * 1.3
        
        return {
            'inner': inner_radius,
            'middle': middle_radius,
            'outer': outer_radius,
            'center_x': avg_x,
            'center_y': avg_y
        }
    
    def detect_rings_hough(self, center_x, center_y, min_radius, max_radius):
        if self.processed_image is None:
            self.preprocess_image()
        
        h, w = self.processed_image.shape[:2]
        y1 = max(0, int(center_y - max_radius - 10))
        y2 = min(h, int(center_y + max_radius + 10))
        x1 = max(0, int(center_x - max_radius - 10))
        x2 = min(w, int(center_x + max_radius + 10))
        
        roi = self.processed_image[y1:y2, x1:x2]
        
        roi_center_x = center_x - x1
        roi_center_y = center_y - y1
        
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius/2,
            param1=80,
            param2=20,
            minRadius=int(min_radius),
            maxRadius=int(max_radius)
        )
        
        if circles is None:
            return self.detect_rings(center_x, center_y, min_radius, max_radius)
        
        circles = np.round(circles[0, :]).astype("int")
        
        filtered_circles = []
        for (x, y, r) in circles:
            dist = np.sqrt((x - roi_center_x)**2 + (y - roi_center_y)**2)
            if dist < min_radius/2:
                filtered_circles.append((x + x1, y + y1, r))
        
        filtered_circles.sort(key=lambda x: x[2])
        
        if len(filtered_circles) < 3:
            return self.detect_rings(center_x, center_y, min_radius, max_radius)
        
        radii = [circle[2] for circle in filtered_circles]
        
        if len(radii) >= 3:
            inner_radius = radii[0]
            middle_radius = radii[len(radii)//2]
            outer_radius = radii[-1]
        else:
            return self.detect_rings(center_x, center_y, min_radius, max_radius)
        
        return {
            'inner': inner_radius,
            'middle': middle_radius,
            'outer': outer_radius
        }
    
    def visualize_detection(self, center_x, center_y, radii):
        """
        Create a visualization of the detected rings.
        
        Args:
            center_x, center_y: Circle center coordinates
            radii: Dictionary with inner, middle, and outer ring radii
            
        Returns:
            Image with visualized rings
        """
        if self.image is None:
            raise ValueError("No image loaded")
        
        vis_image = self.image.copy()
        
        cv2.circle(vis_image, (int(center_x), int(center_y)), 3, (0, 255, 255), -1)
        
        colors = {
            'inner': (255, 0, 0),   
            'middle': (0, 255, 0),  
            'outer': (0, 0, 255)    
        }
        
        for ring_type, radius in radii.items():
            cv2.circle(
                vis_image, 
                (int(center_x), int(center_y)), 
                int(radius), 
                colors[ring_type], 
                2
            )
        
        return vis_image
