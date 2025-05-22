# Image Processor

This document provides a detailed explanation of the Image Processor component implemented in the Zeeman Effect Analysis Software. The Image Processor is responsible for loading, enhancing, and analyzing spectral images to detect spectral lines.

## Overview

The Image Processor module provides essential image processing capabilities for the Zeeman Effect Analysis Software. It handles image loading, enhancement through various filters, and contains the intelligent auto-ring detection algorithm for spectral line detection.

## Class Structure

The `ImageProcessor` class is the main component of this module, providing the following key functionalities:

1. **Image Loading**: Loads images from disk for analysis
2. **Image Enhancement**: Applies various filters to improve image quality for analysis
3. **Spectral Line Detection**: Implements the intelligent auto-ring detection algorithm

## Implementation Details

### 1. Image Loading

The Image Processor loads images from disk using OpenCV:

```python
def load_image(self, file_path):
    self.image = cv2.imread(file_path)
    if self.image is None:
        raise ValueError("Failed to load image")
    return self.image
```

This method loads the image from the specified file path and performs basic validation to ensure the image was loaded successfully.

### 2. Image Enhancement

The Image Processor enhances the loaded image to improve the visibility of spectral lines:

```python
def enhance_image(self):
    if self.image is None:
        raise ValueError("No image loaded")
    
    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    self.processed_image = clahe.apply(blurred)
    
    return self.processed_image
```

The enhancement process involves:

1. **Grayscale Conversion**: Converts the RGB image to grayscale to simplify processing
2. **Gaussian Blur**: Applies a Gaussian blur to reduce noise and smooth the image
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances local contrast to make spectral lines more visible

### 3. Spectral Line Detection

The Image Processor implements the intelligent auto-ring detection algorithm to detect spectral lines:

```python
def detect_spectral_lines(self, processed_image: np.ndarray, initial_center_x: int, initial_center_y: int, radius_lower_limit: int, radius_upper_limit: int, center_search_window_half_size: int = 5) -> Optional[Tuple[int, int, int]]:
    # Input validation
    if not (0 <= radius_lower_limit < radius_upper_limit):
        raise ValueError("Radius limits are invalid.")
    if processed_image is None:
        raise ValueError("Processed image is not available.")
    if processed_image.ndim != 2:
        raise ValueError("Processed image must be grayscale.")
    if center_search_window_half_size < 0:
        raise ValueError("Center search window half size must be non-negative.")

    # Generate candidate centers
    search_grid_size = center_search_window_half_size * 2
    
    candidate_centers = []
    for dx in range(-search_grid_size, search_grid_size + 1, 2):  
        for dy in range(-search_grid_size, search_grid_size + 1, 2):
            candidate_centers.append((initial_center_x + dx, initial_center_y + dy))
    
    if (initial_center_x, initial_center_y) not in candidate_centers:
        candidate_centers.append((initial_center_x, initial_center_y))
    
    weighted_results = {}
    
    # Evaluate each candidate center
    for center_x, center_y in candidate_centers:
        # Create annular mask
        mask = np.zeros(processed_image.shape, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius_upper_limit, 255, -1)
        cv2.circle(mask, (center_x, center_y), radius_lower_limit, 0, -1)
        
        roi_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
        
        # Apply HoughCircles detection
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
        
        # Evaluate detected circles
        if circles is not None:
            detected_circles = circles[0, :]
            
            for circle in detected_circles:
                x, y, r = circle[0], circle[1], circle[2]
                
                # 1. Distance from initial center
                distance_from_initial = np.sqrt((x - initial_center_x)**2 + (y - initial_center_y)**2)
                distance_weight = 1.0 / (1.0 + 0.1 * distance_from_initial)
                
                # 2. Edge strength
                circle_mask = np.zeros(processed_image.shape, dtype=np.uint8)
                cv2.circle(circle_mask, (int(round(x)), int(round(y))), int(round(r)), 255, 1)
                
                edge_pixels = cv2.bitwise_and(processed_image, processed_image, mask=circle_mask)
                non_zero_pixels = edge_pixels[edge_pixels > 0]
                
                if len(non_zero_pixels) > 0:
                    edge_strength = np.mean(non_zero_pixels)
                    edge_weight = edge_strength / 255.0  
                else:
                    edge_weight = 0.0
                
                # 3. Circle completeness
                circle_perimeter = 2 * np.pi * r
                completeness_weight = min(1.0, len(non_zero_pixels) / max(1, circle_perimeter))
                
                # 4. Center proximity
                center_proximity = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                center_weight = 1.0 / (1.0 + center_proximity)
                
                # Calculate final weight
                final_weight = (
                    0.2 * distance_weight +  # Prioritize allowing center to shift
                    0.5 * edge_weight +      # Prioritize strong edges more
                    0.2 * completeness_weight +
                    0.1 * center_weight      # Proximity to candidate center
                )
                
                circle_key = (int(round(x)), int(round(y)), int(round(r)))
                weighted_results[circle_key] = final_weight
    
    # Select the best circle
    if weighted_results:
        best_circle = max(weighted_results.items(), key=lambda item: item[1])[0]
        return best_circle
    
    return None
```

The spectral line detection algorithm consists of several key steps:

1. **Input Validation**: Ensures all input parameters are valid
2. **Candidate Center Generation**: Creates a grid of potential center points around the manually specified center
3. **Circle Detection**: For each candidate center:
   - Creates an annular mask to focus detection on the specified radius range
   - Applies OpenCV's HoughCircles algorithm to detect circles
4. **Quality Evaluation**: For each detected circle:
   - Calculates distance from the initial center (20% weight)
   - Measures edge strength along the circle (50% weight)
   - Evaluates circle completeness (20% weight)
   - Considers proximity to the candidate center (10% weight)
5. **Optimal Selection**: Selects the circle with the highest overall quality score

## Quality Factors Explained

The algorithm evaluates circles based on four key quality factors:

1. **Distance from Initial Center (20%)**: Favors circles with centers close to the manually specified point. A lower weight here allows more flexibility for the center to be adjusted if other quality factors (like edge strength) are high.

2. **Edge Strength (50%)**: Measures the average intensity of pixels along the circle's edge. Stronger edges are a primary indicator of a distinct spectral line and are given the highest weight.

3. **Circle Completeness (20%)**: Evaluates how much of the circle's perimeter is actually detected. More complete circles receive higher weights.

4. **Center Proximity (10%)**: Considers how close the detected circle's center is to the *candidate center point* currently being evaluated. This helps refine the choice among circles found from a specific candidate center.

## Integration with the Measurement Workflow

The Image Processor is integrated with the Measurement Controller, which handles user interactions and delegates to the Image Processor for the actual detection:

```python
detected_info = self.mw.image_processor.detect_spectral_lines(
    enhanced_image, initial_center_x, initial_center_y, 
    lower_rad, upper_rad, center_search_window_half_size=center_search_window_size)
```

## Benefits

The Image Processor module offers several advantages:

1. **Enhanced Image Quality**: The image enhancement process improves the visibility of spectral lines, making them easier to detect.

2. **Accurate Detection**: The intelligent auto-ring detection algorithm optimizes the center point and accurately detects spectral line radii.

3. **Robust Processing**: The module handles various image qualities and provides robust detection even in challenging conditions.

4. **Flexible Configuration**: The detection algorithm can be configured with different parameters to adapt to various experimental setups.
