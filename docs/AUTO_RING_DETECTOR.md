# Auto-Ring Detector Algorithm

This document provides a detailed explanation of the intelligent auto-ring detection algorithm implemented in the Zeeman Effect Analysis Software. The algorithm is designed to optimize the center point of spectral rings and accurately detect their radii, even when the manually specified center point is slightly off.

## Overview

The auto-ring detector uses a weighted multi-point evaluation approach to find the optimal center point and radius for spectral rings. Unlike traditional circle detection methods that rely solely on the manually specified center, our approach evaluates multiple candidate center points around the initial position and selects the one that produces the highest quality circle detection.

## Algorithm

The algorithm consists of several key components:

1. **Candidate Center Generation**: Creates a grid of potential center points around the manually specified center
2. **Circle Detection**: Applies OpenCV's HoughCircles algorithm to each candidate center
3. **Quality Evaluation**: Assigns weights to detected circles based on multiple quality factors
4. **Optimal Selection**: Selects the circle with the highest overall quality score

## Implementation Details

### 1. Candidate Center Generation

The algorithm generates a grid of candidate center points around the manually specified center:

```python
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
```

### 2. Circle Detection for Each Candidate

For each candidate center point, the algorithm creates an annular mask and applies the HoughCircles detection:

```python
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
```

### 3. Quality Evaluation

The algorithm evaluates each detected circle based on multiple quality factors:

```python
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
```

### 4. Final Weight Calculation

The algorithm combines all quality factors into a final weight:

```python
# Calculate final weight as a combination of all factors
# Adjust these coefficients to change the importance of each factor
final_weight = (
    0.2 * distance_weight +  # Prioritize allowing center to shift if other factors are good
    0.5 * edge_weight +      # Prioritize strong edges more
    0.2 * completeness_weight +
    0.1 * center_weight      # Proximity to current candidate center
)

# Store the result with its weight
circle_key = (int(round(x)), int(round(y)), int(round(r)))
weighted_results[circle_key] = final_weight
```

### 5. Optimal Selection

Finally, the algorithm selects the circle with the highest overall quality score:

```python
# Select the circle with the highest weight
if weighted_results:
    best_circle = max(weighted_results.items(), key=lambda item: item[1])[0]
    return best_circle
```

## Quality Factors Explained

The algorithm evaluates circles based on four key quality factors:

1. **Distance from Initial Center (20%)**: Favors circles with centers close to the manually specified point. A lower weight here allows more flexibility for the center to be adjusted if other quality factors (like edge strength) are high.

2. **Edge Strength (50%)**: Measures the average intensity of pixels along the circle's edge. Stronger edges are a primary indicator of a distinct spectral line and are given the highest weight.

3. **Circle Completeness (20%)**: Evaluates how much of the circle's perimeter is actually detected. More complete circles receive higher weights.

4. **Center Proximity (10%)**: Considers how close the detected circle's center is to the *candidate center point* currently being evaluated. This helps refine the choice among circles found from a specific candidate center.

## User Feedback

The algorithm provides detailed feedback to the user about the detection process:

```python
# Calculate the center adjustment distance
center_shift_distance = np.sqrt((det_x - original_center_qpoint.x())**2 + 
                             (det_y - original_center_qpoint.y())**2)

# Create a detailed message with information about the detection
msg = f"Auto-detection for {ring_type_to_update} ring successful:\n"
msg += f"• Detected radius: {det_r:.2f} pixels\n"

if center_shift_distance > 0.5:  # Only mention adjustment if it's significant
    msg += f"• Center point adjusted by {center_shift_distance:.2f} pixels\n"
    msg += f"• Original center: ({original_center_qpoint.x()}, {original_center_qpoint.y()})\n"
    msg += f"• Optimized center: ({det_x}, {det_y})\n"
    msg += "\nThe center was automatically adjusted to better match the spectral ring pattern."
else:
    msg += f"• Center point remained at ({det_x}, {det_y})\n"
    msg += "\nThe manually specified center point was optimal."
```

## Benefits

The intelligent auto-ring detector offers several advantages:

1. **Improved Accuracy**: By optimizing the center point, the algorithm can detect spectral rings more accurately, even when the manually specified center is slightly off.

2. **Robust Detection**: The weighted evaluation approach considers multiple quality factors, making the detection more robust against noise and imperfections in the image.

3. **Detailed Feedback**: The algorithm provides clear feedback about the detection process, helping users understand how and why adjustments were made.

4. **Adaptability**: The algorithm adapts to different image qualities and spectral ring patterns by evaluating multiple factors.

## Integration with the Measurement Workflow

The auto-ring detector is integrated into the measurement workflow through the `MeasurementController` class, which handles user interactions and delegates to the `ImageProcessor` for the actual detection:

```python
# Increase the search window size for better center point adjustment
center_search_window_size = 10  # Larger window for more candidate points

detected_info = self.mw.image_processor.detect_spectral_lines(
    enhanced_image, initial_center_x, initial_center_y, 
    lower_rad, upper_rad, center_search_window_half_size=center_search_window_size)
```

## Conclusion

The intelligent auto-ring detector represents a significant enhancement to the Zeeman Effect Analysis Software, making spectral line measurements more accurate and robust. By evaluating multiple candidate center points and considering various quality factors, the algorithm can optimize the detection process and provide users with more reliable results.
