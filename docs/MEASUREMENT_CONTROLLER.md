# Measurement Controller

This document provides a detailed explanation of the Measurement Controller component implemented in the Zeeman Effect Analysis Software. The Measurement Controller is responsible for coordinating user interactions, managing measurement states, and orchestrating the spectral line detection process.

## Overview

The Measurement Controller serves as a bridge between the user interface and the underlying image processing and physics calculations. It manages the state of measurements, handles user interactions with the image, and coordinates the auto-detection of spectral lines.

## Class Structure

The `MeasurementController` class is the central component for managing measurements, providing the following key functionalities:

1. **Measurement State Management**: Tracks and manages the current measurement state
2. **Mode Control**: Handles different interaction modes (center selection, ring selection, calibration)
3. **Image Click Handling**: Processes user clicks on the image based on the current mode
4. **Auto-Detection Coordination**: Manages the auto-detection process for spectral lines

## Implementation Details

### 1. Initialization and State Management

The Measurement Controller initializes the measurement state and provides methods to reset or initialize new measurements:

```python
def __init__(self, main_window_instance): 
    self.mw = main_window_instance 

    self.current_mode: Optional[str] = None
    self.calibration_points: List[QPoint] = []
    self.current_measurement: Dict[str, Any] = { 
        'center': None, 'type': None, 'radii': {'inner': None, 'middle': None, 'outer': None}
    }

    self.auto_detect_limits: Dict[str, Optional[float]] = {'lower': None, 'upper': None}
    self.is_defining_annulus: bool = False

def reset_all_measurement_states(self):
    self.current_mode = None
    self.calibration_points = []
    self.current_measurement = {
        'center': None, 'type': None, 'radii': {'inner': None, 'middle': None, 'outer': None}
    }
    self.auto_detect_limits = {'lower': None, 'upper': None}
    self.is_defining_annulus = False

def initialize_for_new_measurement(self):
    # Preserves the center point if already set, resets other measurement details.
    # Also resets the current mode and any auto-detection states (annulus).
    current_center = self.current_measurement.get('center') 
    self.current_measurement = {
        'center': current_center, 'type': None, 'radii': {'inner': None, 'middle': None, 'outer': None}
    }
    self.current_mode = None
    self.auto_detect_limits = {'lower': None, 'upper': None}
    self.is_defining_annulus = False
```

The state management includes:

1. **Current Mode**: Tracks the current interaction mode (center, inner, middle, outer, auto_inner, auto_middle, auto_outer, calibrate)
2. **Calibration Points**: Stores points used for spatial calibration
3. **Current Measurement**: Maintains the current measurement state, including center point and radii for inner, middle, and outer rings
4. **Auto-Detect Limits**: Tracks the annulus boundaries for auto-detection
5. **Defining Annulus Flag**: Indicates whether the user is currently defining an annulus for auto-detection

### 2. Mode Control

The Measurement Controller manages different interaction modes:

```python
def set_mode(self, mode: Optional[str]):
    self.current_mode = mode
    if mode == 'center':
        self.initialize_for_new_measurement() 
        self.is_defining_annulus = False
        self.auto_detect_limits = {'lower': None, 'upper': None}
    elif mode and mode.startswith('auto_'):
        if self.current_measurement.get('center') is None:
            QMessageBox.information(self.mw, 'Set Center First', 
                                    'Please set the center point before defining an annulus for auto-detection.')
            self.current_mode = None 
            self.is_defining_annulus = False
            return 
        self.is_defining_annulus = True
        self.auto_detect_limits = {'lower': None, 'upper': None}
    else: 
        self.is_defining_annulus = False
        self.auto_detect_limits = {'lower': None, 'upper': None}
    
    self.mw.update_display()
```

The mode control handles:

1. **Center Mode**: Initializes a new measurement while preserving the center point
2. **Auto-Detection Modes**: Validates that a center point exists and sets up for annulus definition
3. **Other Modes**: Resets auto-detection state

### 3. Image Click Handling

The Measurement Controller processes user clicks on the image based on the current mode:

```python
def handle_image_click(self, pos: QPoint):
    if self.mw.current_image_index < 0 or not self.mw.images:
        QMessageBox.warning(self.mw, "No Image", "Please load an image first.")
        return

    current_image_data = self.mw.images[self.mw.current_image_index]

    if self.is_defining_annulus and self.current_mode and self.current_mode.startswith('auto_'):
        # Handle auto-detection annulus definition
        # ...
    elif self.current_mode == 'calibrate':
        # Handle calibration point selection
        # ...
    elif self.current_mode == 'center':
        # Handle center point selection
        # ...
    elif self.current_mode in ['inner', 'middle', 'outer']:
        # Handle manual ring selection
        # ...
    else:
        pass
```

The click handling includes:

1. **Auto-Detection Annulus Definition**: Processes clicks to define the inner and outer boundaries of the annulus for auto-detection
2. **Calibration**: Collects two points and calculates the mm-per-pixel scale
3. **Center Selection**: Sets the center point for measurements
4. **Manual Ring Selection**: Calculates the radius based on the distance from the center to the clicked point

### 4. Auto-Detection Process

The Measurement Controller coordinates the auto-detection process for spectral lines:

```python
# Inside handle_image_click method, when defining annulus for auto-detection
try:
    raw_rgb_image = current_image_data['image']
    self.mw.image_processor.image = raw_rgb_image 
    enhanced_image = self.mw.image_processor.enhance_image() 
    
    initial_center_x = center_point.x()
    initial_center_y = center_point.y()
    lower_rad = int(self.auto_detect_limits['lower'])
    upper_rad = int(self.auto_detect_limits['upper'])

    center_search_window_size = 10  
    
    detected_info = self.mw.image_processor.detect_spectral_lines(
        enhanced_image, initial_center_x, initial_center_y, 
        lower_rad, upper_rad, center_search_window_half_size=center_search_window_size)

    if detected_info:
        det_x, det_y, det_r = detected_info
        original_center_qpoint = center_point
        new_center_qpoint = QPoint(det_x, det_y)
        
        center_shift_distance = np.sqrt((det_x - original_center_qpoint.x())**2 + 
                                     (det_y - original_center_qpoint.y())**2)
        
        self.current_measurement['center'] = new_center_qpoint
        self.current_measurement['radii'][ring_type_to_update] = det_r
        self.current_measurement['type'] = ring_type_to_update
        
    else:
        self.current_measurement['radii'][ring_type_to_update] = None
        QMessageBox.warning(self.mw, 'Failure', f"Auto-detection failed for {ring_type_to_update} ring.")
except Exception as e:
    if ring_type_to_update: self.current_measurement['radii'][ring_type_to_update] = None
    QMessageBox.critical(self.mw, "Processing Error", f"Error during {ring_type_to_update} ring detection: {str(e)}")
finally:
    self._reset_auto_detect_state_and_update_ui()
```

The auto-detection process involves:

1. **Image Enhancement**: Enhances the image to improve spectral line visibility
2. **Parameter Preparation**: Sets up parameters for the auto-detection algorithm
3. **Detection Execution**: Calls the Image Processor's detect_spectral_lines method
4. **Result Processing**: Updates the measurement state with the detected information
5. **User Feedback**: Generates detailed feedback about the detection process
6. **Error Handling**: Handles potential errors during the detection process
7. **State Reset**: Resets the auto-detection state after completion

### 5. User Feedback

The Measurement Controller provides detailed feedback to the user about the auto-detection process:

```python
msg = f"Auto-detection for {ring_type_to_update} ring successful:\n"
msg += f"• Detected radius: {det_r:.2f} pixels\n"

if center_shift_distance > 0.5: 
    msg += f"• Center point adjusted by {center_shift_distance:.2f} pixels\n"
    msg += f"• Original center: ({original_center_qpoint.x()}, {original_center_qpoint.y()})\n"
    msg += f"• Optimized center: ({det_x}, {det_y})\n"
    msg += "\nThe center was automatically adjusted to better match the spectral ring pattern."
else:
    msg += f"• Center point remained at ({det_x}, {det_y})\n"
    msg += "\nThe manually specified center point was optimal."
    
QMessageBox.information(self.mw, 'Auto-Detection Success', msg)
```

The feedback includes:

1. **Detection Success**: Confirms successful detection of the spectral line
2. **Detected Radius**: Reports the detected radius in pixels
3. **Center Adjustment**: If significant, reports the adjustment made to the center point
4. **Comparison**: Shows the original and optimized center points
5. **Explanation**: Provides context about why the adjustment was made

## Integration with the Main Window

The Measurement Controller is tightly integrated with the Main Window, which provides access to the Image Processor and other components:

```python
def __init__(self, main_window_instance): 
    self.mw = main_window_instance 
    # ...
```

This integration allows the Measurement Controller to:

1. **Access the Image Processor**: For image enhancement and spectral line detection
2. **Update the Display**: Refresh the image display with measurement overlays
3. **Update Measurements**: Update the measurement data in the Main Window
