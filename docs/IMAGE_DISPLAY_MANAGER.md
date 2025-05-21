# Image Display Manager

This document provides a detailed explanation of the Image Display Manager component implemented in the Zeeman Effect Analysis Software. The Image Display Manager is responsible for rendering images, handling zoom operations, and drawing measurement overlays.

## Overview

The Image Display Manager serves as the visual interface between the user and the image data. It handles the display of spectral images, manages zoom operations, and renders visual overlays that represent measurements and calibration points. This component is crucial for providing visual feedback to the user during the measurement process.

## Class Structure

The `ImageDisplayManager` class is the main component of this module, providing the following key functionalities:

1. **Image Display**: Renders images in the UI with proper scaling
2. **Overlay Rendering**: Draws measurement overlays on top of the displayed image
3. **Zoom Operations**: Handles zoom in, zoom out, and reset view operations
4. **Coordinate Mapping**: Maps between screen coordinates and image coordinates

## Implementation Details

### 1. Initialization

The Image Display Manager is initialized with a reference to the image display label and the main window:

```python
def __init__(self, image_display_label: QLabel, main_window_ref):
    self.image_display_label = image_display_label # This is the QLabel for image
    self.main_window = main_window_ref # Reference to MainWindow to access its state
    
    self.scale_factor = 1.0
```

This initialization sets up:
- A reference to the QLabel that will display the image
- A reference to the main window to access its state
- A scale factor for zoom operations (initially set to 1.0)

### 2. Image Display

The Image Display Manager handles the display of images with proper scaling:

```python
def update_display_pixmap(self, q_img: QImage):
    if self.scale_factor != 1.0:
        if q_img is None: return
        scaled_width = int(q_img.width() * self.scale_factor)
        scaled_height = int(q_img.height() * self.scale_factor)
        if scaled_width > 0 and scaled_height > 0:
            q_img = q_img.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            pass # Or log a warning
    
    if q_img:
        self.image_display_label.setPixmap(QPixmap.fromImage(q_img))
    else:
        self.image_display_label.clear()
```

This method:
- Scales the image according to the current scale factor
- Converts the QImage to a QPixmap and sets it on the display label
- Clears the display if no image is provided

### 3. Image Retrieval and Conversion

The Image Display Manager provides methods to retrieve the current image and convert between OpenCV and Qt image formats:

```python
def get_current_cv_image_for_display(self) -> Optional[np.ndarray]:
    """Returns a copy of the current CV image from MainWindow's state for drawing overlays."""
    if not self.main_window.images or self.main_window.current_image_index < 0:
        self.image_display_label.clear() # Clear display if no image
        return None
    img_data = self.main_window.images[self.main_window.current_image_index]
    return img_data['image'].copy()

def convert_cv_to_qimage(self, cv_img: np.ndarray) -> Optional[QImage]:
    if cv_img is None: return None
    
    if cv_img.ndim == 3: # Color image
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        if channel == 3: # Assuming BGR or RGB
            qformat = QImage.Format.Format_RGB888
        else: # e.g. RGBA
            qformat = QImage.Format.Format_RGBA8888 # Or other appropriate format
    elif cv_img.ndim == 2: # Grayscale image
        height, width = cv_img.shape
        bytes_per_line = width
        qformat = QImage.Format.Format_Grayscale8
    else:
        return None # Unsupported image format

    return QImage(cv_img.data, width, height, bytes_per_line, qformat)
```

These methods:
- Retrieve the current image from the main window's state
- Convert between OpenCV's numpy array format and Qt's QImage format
- Handle different image formats (color, grayscale)

### 4. Overlay Rendering

The Image Display Manager renders measurement overlays on top of the displayed image:

```python
def redraw_image_with_overlays(self):
    display_cv_img = self.get_current_cv_image_for_display()
    if display_cv_img is None:
        self.main_window.update_navigation() 
        return

    mw = self.main_window # Alias for brevity
    mc = mw.measurement_controller # Alias for MeasurementController
    
    # Draw calibration points
    if mc.calibration_points: # Check if list is not empty
        for point in mc.calibration_points:
            cv2.circle(display_cv_img, (point.x(), point.y()), 3, (0, 255, 0), -1)

    # Draw center point and rings
    if mc.current_measurement and mc.current_measurement.get('center') is not None:
        center_qpoint = mc.current_measurement['center']
        center_coords = (center_qpoint.x(), center_qpoint.y())
        cv2.circle(display_cv_img, center_coords, 3, (255, 0, 0), -1) 

        manual_colors = {'inner': (0, 0, 255), 'middle': (0, 255, 0), 'outer': (255, 0, 0)}
        if mc.current_measurement.get('radii'):
            for radius_type, radius_pixels in mc.current_measurement['radii'].items():
                if radius_pixels is not None:
                    color = manual_colors.get(radius_type, (255, 255, 0))
                    cv2.circle(display_cv_img, center_coords, int(radius_pixels), color, 1)
        
        # Draw auto-detection annulus
        if mc.auto_detect_limits and mc.auto_detect_limits.get('lower') is not None and mc.auto_detect_limits.get('upper') is not None:
            cv2.circle(display_cv_img, center_coords, int(mc.auto_detect_limits['lower']), (255, 255, 0), 1) # Cyan
            cv2.circle(display_cv_img, center_coords, int(mc.auto_detect_limits['upper']), (0, 255, 255), 1)  # Yellow
        elif mc.is_defining_annulus and mc.auto_detect_limits and mc.auto_detect_limits.get('lower') is not None:
            cv2.circle(display_cv_img, center_coords, int(mc.auto_detect_limits['lower']), (255, 165, 0), 1) # Orange
    
    q_img = self.convert_cv_to_qimage(display_cv_img)
    self.update_display_pixmap(q_img) # This now handles scaling and setting pixmap
    
    mw.update_navigation()
```

This method:
- Retrieves the current image for display
- Draws calibration points as small green circles
- Draws the center point as a small red circle
- Draws measured rings with different colors (blue for inner, green for middle, red for outer)
- Draws the auto-detection annulus with appropriate colors
- Converts the image with overlays to a QImage and updates the display
- Updates the navigation controls

### 5. Zoom Operations

The Image Display Manager handles zoom operations:

```python
def zoom_in(self):
    self.scale_factor *= 1.2
    self.redraw_image_with_overlays()

def zoom_out(self):
    self.scale_factor /= 1.2
    self.redraw_image_with_overlays()

def reset_view(self):
    self.scale_factor = 1.0
    self.redraw_image_with_overlays()
```

These methods:
- Adjust the scale factor for zoom in (multiply by 1.2) and zoom out (divide by 1.2)
- Reset the scale factor to 1.0 for reset view
- Redraw the image with overlays after adjusting the scale factor

### 6. Coordinate Mapping

The Image Display Manager maps between screen coordinates and image coordinates:

```python
def get_image_coordinates(self, event_pos: QPoint) -> Optional[QPoint]:
    pixmap = self.image_display_label.pixmap()
    if not pixmap or pixmap.isNull(): return None # Check for null pixmap
        
    label_size = self.image_display_label.size()
    pixmap_size = pixmap.size() # Actual displayed pixmap size after scaling by Qt
    
    # Calculate offsets for centered pixmap
    x_offset = (label_size.width() - pixmap_size.width()) / 2
    y_offset = (label_size.height() - pixmap_size.height()) / 2
    
    # Calculate position relative to pixmap
    x_rel_pixmap = event_pos.x() - x_offset
    y_rel_pixmap = event_pos.y() - y_offset

    # Get original image dimensions
    if not self.main_window.images or self.main_window.current_image_index < 0: return None
    original_img_data = self.main_window.images[self.main_window.current_image_index]['image']
    original_height, original_width = original_img_data.shape[:2]

    if pixmap_size.width() == 0 or pixmap_size.height() == 0: return None # Avoid division by zero

    # Map from pixmap coordinates to original image coordinates
    original_x = (x_rel_pixmap / pixmap_size.width()) * original_width
    original_y = (y_rel_pixmap / pixmap_size.height()) * original_height
    
    # Ensure coordinates are within image bounds
    x_coord = max(0, min(original_width - 1, int(original_x)))
    y_coord = max(0, min(original_height - 1, int(original_y)))
    
    return QPoint(x_coord, y_coord)
```

This method:
- Takes a screen position (from a mouse event) and converts it to image coordinates
- Accounts for the image being centered in the label
- Scales the coordinates based on the ratio between the displayed pixmap and the original image
- Ensures the coordinates are within the bounds of the original image
- Returns the corresponding point in the original image coordinates

## Color Coding

The Image Display Manager uses a consistent color coding scheme for overlays:

1. **Calibration Points**: Green circles
2. **Center Point**: Red circle
3. **Inner Ring**: Blue circle
4. **Middle Ring**: Green circle
5. **Outer Ring**: Red circle
6. **Auto-Detection Lower Limit**: Cyan circle (when complete) or Orange circle (when defining)
7. **Auto-Detection Upper Limit**: Yellow circle

This color coding helps users distinguish between different types of measurements and provides visual feedback during the measurement process.

## Integration with the Main Window

The Image Display Manager is tightly integrated with the Main Window and Measurement Controller:

```python
def __init__(self, image_display_label: QLabel, main_window_ref):
    self.image_display_label = image_display_label
    self.main_window = main_window_ref
    # ...
```

This integration allows the Image Display Manager to:
- Access the current image from the Main Window
- Access the measurement state from the Measurement Controller
- Update the display based on changes in the measurement state
- Map user interactions (clicks) to image coordinates

## Benefits

The Image Display Manager offers several advantages:

1. **Visual Feedback**: Provides clear visual feedback about measurements and calibration
2. **Interactive Visualization**: Allows users to zoom in and out to examine details
3. **Coordinate Mapping**: Accurately maps between screen coordinates and image coordinates
4. **Consistent Rendering**: Ensures consistent rendering of overlays with appropriate colors
5. **Efficient Updates**: Updates the display efficiently when the measurement state changes
