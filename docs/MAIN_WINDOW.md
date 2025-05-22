# Main Window

This document provides a detailed explanation of the Main Window component implemented in the Zeeman Effect Analysis Software. The Main Window serves as the central hub of the application, coordinating all other components and providing the primary user interface.

## Overview

The Main Window is the core component of the Zeeman Effect Analysis Software, providing the main user interface and coordinating the interactions between different components. It manages the application state, handles user interactions, and orchestrates the workflow for measuring and analyzing the Zeeman effect.

## Class Structure

The `MainWindow` class is the central component of the application, providing the following key functionalities:

1. **UI Management**: Creates and manages the user interface elements
2. **Image Management**: Handles loading, displaying, and navigating between images
3. **Measurement Coordination**: Coordinates the measurement process
4. **Data Management**: Manages measurement data and results
5. **Component Integration**: Integrates various components such as the Image Processor, Measurement Controller, and Image Display Manager

## Implementation Details

### 1. Initialization

The Main Window initializes the application and sets up all necessary components:

```python
def __init__(self):
    super().__init__()
    
    # Initialize state variables for image management and calibration
    self.images = []
    self.current_image_index = -1
    self.calibration_distance_mm = 10.0 # Default known distance for calibration
    self.measurements = [] # Stores completed MeasurementData objects
    
    # Initialize components
    self.image_processor = ImageProcessor()
    # MeasurementController now manages all measurement-specific states like mode, current points, annulus etc.
    self.measurement_controller = MeasurementController(self)
    self.image_display_manager = ImageDisplayManager(self.image_display, self, self.ui_manager) # Assuming image_display and ui_manager are set up in setup_ui
    
    # Set up UI (details in setup_ui method)
    self.ui_manager = UIManager(self)
    self.ui_manager.setup_layout() # This will set up self.image_display etc.
    
    # Initialize other windows (Plot, Table, Results, Calibration)
    self.plot_window = PlotWindow(self.ui_manager)
    self.table_window = TableWindow(self.ui_manager)
    self.results_window = ResultsWindow(self.ui_manager)
    self.calibration_window = CalibrationWindow(self)

    # Connect signals and slots
    self.connect_signals() # Placeholder for actual signal connections
```

The initialization process:
- Sets up state variables for managing loaded images, the current image, calibration scale, and collected measurements.
- Creates instances of key components: `ImageProcessor` for image operations, `MeasurementController` (which now encapsulates all measurement-related state and logic), `ImageDisplayManager` for rendering, and `UIManager` for UI construction.
- Initializes various pop-up windows for displaying plots, tables, results, and handling magnetic field calibration.
- Sets up the user interface via the `UIManager`.
- Connects signals and slots for event handling (specific connections are typically in `connect_signals` or UI setup).

### 2. UI Setup

The Main Window sets up a comprehensive user interface:

```python
def setup_ui(self):
    # Set window properties
    self.setWindowTitle("Zeeman Effect Analysis")
    self.resize(1200, 800)
    
    # Create central widget and layout
    self.central_widget = QWidget()
    self.setCentralWidget(self.central_widget)
    self.main_layout = QHBoxLayout(self.central_widget)
    
    # Create image display area
    self.setup_image_display()
    
    # Create control panel
    self.setup_control_panel()
    
    # Create menu bar
    self.setup_menu_bar()
    
    # Create status bar
    self.statusBar().showMessage("Ready")
```

The UI setup includes:
- Setting window properties
- Creating the central widget and main layout
- Setting up the image display area
- Setting up the control panel with measurement tools
- Setting up the menu bar with file and analysis options
- Setting up the status bar for user feedback

### 3. Image Management

The Main Window provides methods for loading, displaying, and navigating between images:

```python
def load_image(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    )
    
    if file_path:
        try:
            # Load image using OpenCV
            cv_image = self.image_processor.load_image(file_path)
            
            # Create image data dictionary
            image_data = {
                'file_path': file_path,
                'image': cv_image,
                'B_field': 0.0,
                'wavelength': 589.3e-9,  # Default: sodium D-line
                'mm_per_pixel': 0.1,     # Default calibration
                'measurement': None
            }
            
            # Add to image list and update display
            self.images.append(image_data)
            self.current_image_index = len(self.images) - 1
            self.update_display()
            self.update_navigation()
            
            # Reset measurement state
            self.measurement_controller.reset_all_measurement_states()
            
            # Update status
            self.statusBar().showMessage(f"Loaded: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
```

The image management includes:
- Loading images from disk using the Image Processor
- Creating a data structure to store image metadata
- Updating the display and navigation controls
- Resetting the measurement state for the new image
- Providing user feedback through the status bar

### 4. Measurement Coordination

The Main Window coordinates the measurement process by delegating to the `MeasurementController`. It primarily acts as a passthrough for setting modes and handling image clicks, with the `MeasurementController` managing the actual state and logic.

```python
def set_measurement_mode(self, mode):
    self.measurement_controller.set_mode(mode)
    
    # Update status bar based on the mode set in MeasurementController
    if self.measurement_controller.current_mode == 'center':
        self.statusBar().showMessage("Click to set center point")
    elif self.measurement_controller.current_mode == 'calibrate':
        self.statusBar().showMessage("Click to set calibration points (2 points needed)")
    elif self.measurement_controller.current_mode in ['inner', 'middle', 'outer']:
        self.statusBar().showMessage(f"Click to set {self.measurement_controller.current_mode} ring radius")
    elif self.measurement_controller.current_mode and self.measurement_controller.current_mode.startswith('auto_'):
        ring_type = self.measurement_controller.current_mode.split('_')[1]
        self.statusBar().showMessage(f"Click to define annulus for auto-detecting {ring_type} ring")
    else:
        self.statusBar().showMessage("Ready") # Default or cleared mode
```

The measurement coordination includes:
- Relaying the desired measurement mode to the `MeasurementController`.
- The `MeasurementController` then updates its internal state (e.g., `current_mode`, `is_defining_annulus`).
- `MainWindow` updates the status bar to guide the user based on the state now held within `MeasurementController`.
- User interactions on the image (clicks) are passed to `MeasurementController.handle_image_click` which processes them based on its current state.

### 5. Display Update

The Main Window updates the display through the Image Display Manager:

```python
def update_display(self):
    if not self.images or self.current_image_index < 0:
        self.image_display_manager.redraw_image_with_overlays()
        return
    
    self.image_display_manager.redraw_image_with_overlays()
    
    # Update metadata display
    self.update_metadata_display()
    
    # Update measurements display
    self.update_measurements_display()
```

The display update includes:
- Redrawing the image with overlays through the Image Display Manager
- Updating the metadata display with current image information
- Updating the measurements display with current measurement data

### 6. Metadata Management

The Main Window manages metadata for each image:

```python
def update_metadata_display(self):
    if not self.images or self.current_image_index < 0:
        self.metadata_display.clear()
        return
    
    current_image_data = self.images[self.current_image_index]
    
    metadata_text = f"File: {os.path.basename(current_image_data['file_path'])}\n"
    metadata_text += f"B-field: {current_image_data['B_field']:.4f} T\n"
    metadata_text += f"Wavelength: {current_image_data['wavelength']*1e9:.2f} nm\n"
    metadata_text += f"Scale: {current_image_data['mm_per_pixel']*1000:.4f} mm/pixel"
    
    self.metadata_display.setText(metadata_text)
```

The metadata management includes:
- Displaying the file name, magnetic field strength, wavelength, and scale
- Formatting the values for clear presentation
- Updating the display when the current image changes

### 7. Measurement Data Management

The Main Window manages measurement data for each image:

```python
def update_measurements_display(self):
    if not self.images or self.current_image_index < 0:
        self.measurements_display.clear()
        return
    
    current_image_data = self.images[self.current_image_index]
    mc = self.measurement_controller
    
    measurements_text = "Measurements:\n"
    
    if mc.current_measurement and mc.current_measurement.get('center') is not None:
        center = mc.current_measurement['center']
        measurements_text += f"Center: ({center.x()}, {center.y()})\n"
        
        if mc.current_measurement.get('radii'):
            for radius_type, radius_pixels in mc.current_measurement['radii'].items():
                if radius_pixels is not None:
                    radius_mm = radius_pixels * current_image_data['mm_per_pixel']
                    measurements_text += f"{radius_type.capitalize()} radius: {radius_pixels:.2f} px ({radius_mm:.4f} mm)\n"
    else:
        measurements_text += "No measurements yet"
    
    self.measurements_display.setText(measurements_text)
```

The measurement data management includes:
- Displaying the center point coordinates
- Displaying the radii for inner, middle, and outer rings
- Converting pixel measurements to millimeters using the calibration scale
- Updating the display when measurements change

### 8. Results Calculation

The Main Window provides methods for calculating and displaying results:

```python
def calculate_results(self):
    if not self.images:
        QMessageBox.warning(self, "No Data", "No images loaded for analysis.")
        return
    
    # Create a list of measurements
    measurements = []
    for img_data in self.images:
        if 'measurement' in img_data and img_data['measurement'] is not None:
            # Create and process a ZeemanMeasurement object
            measurement = zeeman.ZeemanMeasurement(
                B_field=img_data['B_field'],
                wavelength=img_data['wavelength'],
                R_center=img_data['measurement']['radii']['middle'],
                R_inner=img_data['measurement']['radii']['inner'],
                R_outer=img_data['measurement']['radii']['outer']
            )
            measurement = zeeman.process_measurement(measurement)
            measurements.append(measurement)
    
    if not measurements:
        QMessageBox.warning(self, "No Measurements", "No valid measurements found.")
        return
    
    # Calculate Bohr magneton
    results = zeeman.calculate_bohr_magneton(measurements)
    
    # Display results
    self.show_results_window(measurements, results)
```

The results calculation includes:
- Creating ZeemanMeasurement objects from the image data
- Processing the measurements to calculate physical quantities
- Calculating the Bohr magneton from the measurements
- Displaying the results in a dedicated window

### 9. Event Handling

The Main Window handles various events, including mouse clicks on the image:

```python
def handle_image_click(self, event):
    if not self.images or self.current_image_index < 0:
        return
    
    # Convert event position to image coordinates
    image_pos = self.image_display_manager.get_image_coordinates(event.pos())
    if image_pos is None:
        return
    
    # Delegate to measurement controller
    self.measurement_controller.handle_image_click(image_pos)
```

The event handling includes:
- Converting event positions to image coordinates
- Delegating to the Measurement Controller for processing
- Updating the display based on the results

## Component Integration

The Main Window integrates various components of the Zeeman Effect Analysis Software:

1. **Image Processor**: For loading and enhancing images
2. **Measurement Controller**: For managing the measurement process
3. **Image Display Manager**: For displaying images and overlays
4. **Zeeman Physics Module**: For calculating physical quantities and results

This integration allows the components to work together seamlessly, providing a comprehensive solution for analyzing the Zeeman effect.

## Workflow Management

The Main Window manages the overall workflow of the application:

1. **Image Loading**: Load spectral images for analysis
2. **Calibration**: Set the scale by measuring a known distance
3. **Center Selection**: Set the center point for measurements
4. **Ring Measurement**: Measure the radii of spectral rings (manually or using auto-detection)
5. **Metadata Entry**: Enter magnetic field strength and wavelength
6. **Results Calculation**: Calculate physical quantities and the Bohr magneton
7. **Results Visualization**: Display results in tables and plots

This workflow guides users through the process of analyzing the Zeeman effect, from image loading to results visualization.

## Benefits

The Main Window offers several advantages:

1. **Centralized Control**: Provides a single point of control for the entire application
2. **Intuitive Interface**: Offers a clear and intuitive user interface for all operations
3. **Comprehensive Workflow**: Guides users through the complete analysis process
4. **Flexible Integration**: Integrates various components into a cohesive whole
5. **Detailed Feedback**: Provides detailed feedback at each step of the process
