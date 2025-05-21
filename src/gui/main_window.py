import csv
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea,
    QGroupBox, QMessageBox, QInputDialog, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QApplication
)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QScreen
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence
import cv2
import numpy as np
from pathlib import Path
from src.physics.zeeman import ZeemanMeasurement, process_measurement, calculate_bohr_magneton
import matplotlib.pyplot as plt
from src.gui.plot_window import PlotWindow
from src.gui.table_window import TableWindow
from src.gui.results_window import ResultsWindow
from src.gui.calibration_window import CalibrationWindow
from src.processing.image_processor import ImageProcessor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Zeeman Effect Analysis')
        
        # Get screen size and set window size
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        
        # Set window size to 80% of screen size
        window_width = int(self.screen_width * 0.8)
        window_height = int(self.screen_height * 0.8)
        
        # Center the window
        x = (self.screen_width - window_width) // 2
        y = (self.screen_height - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        
        # Add keyboard shortcut for test data
        self.shortcut_test = QShortcut(QKeySequence('Ctrl+T'), self)
        self.shortcut_test.activated.connect(self.fill_test_data)
        
        # Measurement variables
        self.images = []  # List of loaded images with their measurements
        self.current_image_index = -1
        self.current_measurement = None
        self.current_mode = None
        self.scale_factor = 1.0
        
        # Image Processor
        self.image_processor = ImageProcessor()

        # Initialize measurement state
        self.current_mode = None  # 'calibrate', 'center', 'inner', 'middle', 'outer', 'auto_inner', 'auto_middle', 'auto_outer'
        self.calibration_points = []
        self.current_measurement = {
            'center': None,
            'type': None,
            'radii': {'inner': None, 'middle': None, 'outer': None}
        }
        self.mm_per_pixel = None
        self.calibration_distance_mm = 10.0  # Default calibration distance

        # State for auto-detection annulus definition
        self.auto_detect_limits = {'lower': None, 'upper': None}
        self.is_defining_annulus = False
        
        # Initialize physics measurements
        self.measurements = []  # List of ZeemanMeasurement objects
        
        # Create calibration window
        self.calibration_window = CalibrationWindow()
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Image navigation controls
        nav_layout = QHBoxLayout()
        self.prev_image_btn = QPushButton('Previous Image')
        self.prev_image_btn.clicked.connect(self.previous_image)
        self.next_image_btn = QPushButton('Next Image')
        self.next_image_btn.clicked.connect(self.next_image)
        self.image_label = QLabel('No image loaded')
        
        nav_layout.addWidget(self.prev_image_btn)
        nav_layout.addWidget(self.image_label)
        nav_layout.addWidget(self.next_image_btn)
        main_layout.addLayout(nav_layout)
        
        # Create horizontal layout for image display and controls
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Image display in a scroll area
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.mousePressEvent = self.image_clicked
        image_container_layout.addWidget(self.image_display)
        image_container_layout.addStretch()
        
        image_scroll.setWidget(image_container)
        content_layout.addWidget(image_scroll, 60)  # 75% of width
        content_layout.setStretch(0, 60)  # Image area stretch factor
        
        # Create control panel with scroll area
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        # Set control panel width to 1/4 of window width
        control_width = int(self.width() * 0.4)
        self.control_scroll.setFixedWidth(control_width)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # Image controls
        image_group = QGroupBox('Image Controls')
        image_layout = QVBoxLayout()
        
        # Load image button
        load_btn = QPushButton('Load Image')
        load_btn.clicked.connect(self.load_image)
        image_layout.addWidget(load_btn)
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton('Zoom In')
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn = QPushButton('Zoom Out')
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_view_btn = QPushButton('Reset View')
        reset_view_btn.clicked.connect(self.reset_view)
        
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(reset_view_btn)
        image_layout.addLayout(zoom_layout)
        
        image_group.setLayout(image_layout)
        control_layout.addWidget(image_group)
        
        # Calibration controls
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        
        calibrate_btn = QPushButton("Calibrate Scale")
        calibrate_btn.clicked.connect(lambda: self.set_measurement_mode('calibrate'))
        calibration_layout.addWidget(calibrate_btn)
        
        self.calibration_label = QLabel("Scale: Not calibrated")
        calibration_layout.addWidget(self.calibration_label)
        
        calibration_group.setLayout(calibration_layout)
        control_layout.addWidget(calibration_group)
        
        # Measurement controls
        measurement_group = QGroupBox("Measurement Controls")
        measurement_layout = QVBoxLayout(measurement_group)
        
        center_btn = QPushButton("Set Center Point")
        center_btn.clicked.connect(lambda: self.set_measurement_mode('center'))
        measurement_layout.addWidget(center_btn)
        
        radius_layout = QHBoxLayout()
        inner_btn = QPushButton("Measure Inner")
        inner_btn.clicked.connect(lambda: self.set_measurement_mode('inner'))
        middle_btn = QPushButton("Measure Middle")
        middle_btn.clicked.connect(lambda: self.set_measurement_mode('middle'))
        outer_btn = QPushButton("Measure Outer")
        outer_btn.clicked.connect(lambda: self.set_measurement_mode('outer'))
        
        radius_layout.addWidget(inner_btn)
        radius_layout.addWidget(middle_btn)
        radius_layout.addWidget(outer_btn)
        measurement_layout.addLayout(radius_layout)

        # Auto-detection buttons
        auto_detect_label = QLabel("Auto Detect Radii (Define Annulus):")
        measurement_layout.addWidget(auto_detect_label)
        auto_radius_layout = QHBoxLayout()
        auto_inner_btn = QPushButton("Auto Detect Inner")
        auto_inner_btn.clicked.connect(lambda: self.set_measurement_mode('auto_inner'))
        auto_middle_btn = QPushButton("Auto Detect Middle")
        auto_middle_btn.clicked.connect(lambda: self.set_measurement_mode('auto_middle'))
        auto_outer_btn = QPushButton("Auto Detect Outer")
        auto_outer_btn.clicked.connect(lambda: self.set_measurement_mode('auto_outer'))

        auto_radius_layout.addWidget(auto_inner_btn)
        auto_radius_layout.addWidget(auto_middle_btn)
        auto_radius_layout.addWidget(auto_outer_btn)
        measurement_layout.addLayout(auto_radius_layout)
        
        reset_btn = QPushButton("Reset Measurements")
        reset_btn.clicked.connect(self.reset_measurements)
        measurement_layout.addWidget(reset_btn)
        
        measurement_group.setLayout(measurement_layout)
        control_layout.addWidget(measurement_group)
        
        # Measurements display
        measurements_group = QGroupBox("Measurements")
        measurements_layout = QVBoxLayout(measurements_group)
        
        # Create table for measurements
        self.measurements_table = QTableWidget()
        self.measurements_table.setColumnCount(3)
        self.measurements_table.setHorizontalHeaderLabels(['Current (A)', 'B Field (T)', 'Actions'])
        self.measurements_table.horizontalHeader().setStretchLastSection(True)
        measurements_layout.addWidget(self.measurements_table)
        
        measurements_group.setLayout(measurements_layout)
        control_layout.addWidget(measurements_group)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        # Parameters layout
        params_layout = QHBoxLayout()
        
        # Current and calibration layout
        current_group = QGroupBox('Magnetic Field')
        current_group_layout = QVBoxLayout()
        
        # Current input
        current_layout = QHBoxLayout()
        current_label = QLabel('Current (A):')
        self.current_input = QDoubleSpinBox()
        self.current_input.setRange(0, 100)
        self.current_input.setDecimals(3)
        self.current_input.setSingleStep(0.1)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_input)
        current_group_layout.addLayout(current_layout)
        
        # Calibration button
        self.calibrate_btn = QPushButton('Calibrate Field')
        self.calibrate_btn.clicked.connect(self.show_calibration)
        current_group_layout.addWidget(self.calibrate_btn)
        
        current_group.setLayout(current_group_layout)
        params_layout.addWidget(current_group)
        
        # Wavelength input
        wavelength_layout = QVBoxLayout()
        wavelength_label = QLabel('Wavelength (nm):')
        self.wavelength_input = QDoubleSpinBox()
        self.wavelength_input.setRange(300, 1000)
        self.wavelength_input.setDecimals(1)
        self.wavelength_input.setValue(643.8)  # Default wavelength
        wavelength_layout.addWidget(wavelength_label)
        wavelength_layout.addWidget(self.wavelength_input)
        params_layout.addLayout(wavelength_layout)
        
        results_layout.addLayout(params_layout)
        
        # Save measurement button
        self.save_measurement_btn = QPushButton('Save Measurement')
        self.save_measurement_btn.clicked.connect(self.save_measurement)
        results_layout.addWidget(self.save_measurement_btn)
        
        # Results buttons layout
        buttons_layout = QHBoxLayout()
        
        # Calculate button
        self.calculate_btn = QPushButton('Calculate Results')
        self.calculate_btn.clicked.connect(self.calculate_results)
        buttons_layout.addWidget(self.calculate_btn)
        
        # Show plot button
        self.show_plot_btn = QPushButton('Show Plot')
        self.show_plot_btn.clicked.connect(self.show_plot)
        buttons_layout.addWidget(self.show_plot_btn)
        
        # Show table button
        self.show_table_btn = QPushButton('Show Data Table')
        self.show_table_btn.clicked.connect(self.show_table)
        buttons_layout.addWidget(self.show_table_btn)
        
        # Show results button
        self.show_results_btn = QPushButton('Show Results')
        self.show_results_btn.clicked.connect(self.show_results)
        buttons_layout.addWidget(self.show_results_btn)
        
        results_layout.addLayout(buttons_layout)
        
        # Export button
        self.export_btn = QPushButton('Export to CSV')
        self.export_btn.clicked.connect(self.export_to_csv)
        results_layout.addWidget(self.export_btn)
        
        results_group.setLayout(results_layout)
        
        # Initialize windows
        self.plot_window = PlotWindow()
        self.table_window = TableWindow()
        self.results_window = ResultsWindow()
        # Add control panel to content layout
        control_layout.addWidget(results_group)
        control_layout.addStretch()
        
        self.control_scroll.setWidget(control_panel)
        content_layout.addWidget(self.control_scroll, 25)  # 25% of width
        content_layout.setStretch(1, 25)  # Control panel stretch factor
        
        # Set the central widget
        self.setCentralWidget(central_widget)
        
        # Update UI state
        self.update_navigation()
    
    def update_display(self):
        if not self.images or self.current_image_index < 0:
            self.image_display.clear()
            return

        # Get current image and create a copy for drawing
        img_data = self.images[self.current_image_index]
        display_img = img_data['image'].copy()

        # Draw calibration points
        for point in self.calibration_points:
            cv2.circle(display_img, (point.x(), point.y()), 3, (0, 255, 0), -1)

        # Draw center point and radii
        if self.current_measurement['center'] is not None:
            center = self.current_measurement['center']
            cv2.circle(display_img, (center.x(), center.y()), 3, (255, 0, 0), -1) # Red dot for center

            # Draw manually set radii with different colors
            manual_colors = {
                'inner': (0, 0, 255),   # Blue
                'middle': (0, 255, 0),  # Green
                'outer': (255, 0, 0)    # Red
            }
            
            for radius_type, radius in self.current_measurement['radii'].items():
                if radius is not None:
                    color = manual_colors.get(radius_type, (255, 255, 0)) # Default to Yellow
                    cv2.circle(display_img, (center.x(), center.y()), int(radius), color, 1)

            # Draw annulus definition feedback
            center_coords = (center.x(), center.y())
            
            # If both auto_detect_limits are set (e.g., after 2nd click and before processing clears them in the finally block),
            # draw them in specified colors: Cyan for lower, Yellow for upper. This takes precedence.
            if self.auto_detect_limits['lower'] is not None and self.auto_detect_limits['upper'] is not None:
                cv2.circle(display_img, center_coords, int(self.auto_detect_limits['lower']), (255, 255, 0), 1) # Cyan
                cv2.circle(display_img, center_coords, int(self.auto_detect_limits['upper']), (0, 255, 255), 1) # Yellow
            # Else, if actively defining the annulus (is_defining_annulus is True) and only the lower limit is set,
            # draw it in orange for active feedback.
            elif self.is_defining_annulus and self.auto_detect_limits['lower'] is not None: # Only lower is set and we are in definition mode
                cv2.circle(display_img, center_coords, int(self.auto_detect_limits['lower']), (255, 165, 0), 1) # Orange
            
        # Convert to QImage and display
        height, width, channel = display_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale image if needed
        if self.scale_factor != 1.0:
            scaled_width = int(width * self.scale_factor)
            scaled_height = int(height * self.scale_factor)
            q_img = q_img.scaled(scaled_width, scaled_height)

        self.image_display.setPixmap(QPixmap.fromImage(q_img))
        self.update_navigation()
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Open Image',
            '',
            'Image Files (*.png *.jpg *.jpeg *.bmp)'
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.critical(self, 'Error', 'Failed to load image')
                return
            
            # Convert BGR to RGB for display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Add new image with its own calibration and measurements
            self.images.append({
                'image': image,
                'calibration_points': [],
                'mm_per_pixel': None,
                'measurement': None
            })
            
            self.current_image_index = len(self.images) - 1
            self.initialize_measurement()
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def reset_measurements(self):
        """Reset all measurements for the current image."""
        if self.current_image_index >= 0:
            # self.images[self.current_image_index]['measurement'] = None # Keep existing measurements
            self.initialize_measurement() # Resets current_measurement, not all image measurements
            self.is_defining_annulus = False
            self.auto_detect_limits = {'lower': None, 'upper': None}
            self.current_mode = None # Stop any active measurement mode
            self.update_display()
            self.update_measurements_display() # Update display based on current state
    
    def get_image_coordinates(self, event_pos):
        """Convert screen coordinates to image coordinates."""
        # Get the scroll area's viewport widget (the one that actually contains the image)
        viewport = self.image_display.parent()
        
        # Get the position of the image label relative to the viewport
        image_pos = self.image_display.pos()
        
        # Get the actual image size from the pixmap
        pixmap = self.image_display.pixmap()
        if not pixmap:
            return None
            
        # Calculate the offset within the image label where the image is drawn
        label_size = self.image_display.size()
        pixmap_size = pixmap.size()
        
        # Calculate centering offsets
        x_offset = (label_size.width() - pixmap_size.width()) / 2
        y_offset = (label_size.height() - pixmap_size.height()) / 2
        
        # Get click position relative to the image label
        pos = event_pos
        
        # Adjust for image position and centering
        x = pos.x() - x_offset
        y = pos.y() - y_offset
        
        # Scale coordinates based on image scaling
        if self.scale_factor != 1.0:
            x = x / self.scale_factor
            y = y / self.scale_factor
        
        # Ensure coordinates are within image bounds
        img_data = self.images[self.current_image_index]
        height, width = img_data['image'].shape[:2]
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        
        return QPoint(x, y)
    
    def image_clicked(self, event):
        if not self.images or self.current_image_index < 0:
            return

        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return

        if self.is_defining_annulus and self.current_mode and self.current_mode.startswith('auto_'):
            if self.current_measurement['center'] is None:
                QMessageBox.warning(self, "Error", "Please set the center point before defining an annulus.")
                self.is_defining_annulus = False
                self.current_mode = None
                return

            dx = pos.x() - self.current_measurement['center'].x()
            dy = pos.y() - self.current_measurement['center'].y()
            clicked_radius = (dx**2 + dy**2)**0.5

            if self.auto_detect_limits['lower'] is None:
                self.auto_detect_limits['lower'] = clicked_radius
                # Optionally, update a status bar: self.statusBar().showMessage("Click to set upper annulus limit.")
                print(f"Lower annulus limit set: {clicked_radius}") # Placeholder for status update
            else:
                self.auto_detect_limits['upper'] = clicked_radius
                if self.auto_detect_limits['lower'] > self.auto_detect_limits['upper']:
                    # Swap them
                    self.auto_detect_limits['lower'], self.auto_detect_limits['upper'] = self.auto_detect_limits['upper'], self.auto_detect_limits['lower']
                
                print(f"Upper annulus limit set: {self.auto_detect_limits['upper']}. Annulus defined: {self.auto_detect_limits}") # Placeholder
                
                # Immediately update display to show both defined annulus limits (cyan and yellow as per update_display logic)
                self.update_display() 

                ring_type_to_update = None # Initialize for broader scope (finally block)
                
                # Defensive check for current_mode and extraction of ring_type_to_update
                if self.current_mode and self.current_mode.startswith('auto_'):
                    ring_type_to_update = self.current_mode.split('_')[1]
                else:
                    # This state should ideally not be reached if set_measurement_mode and other logic is correct.
                    QMessageBox.critical(self, "Internal Error", 
                                         f"Invalid or missing mode for detection: '{self.current_mode}'. Aborting auto-detection.")
                    # Reset state comprehensively and exit this click handling
                    self.current_mode = None
                    self.auto_detect_limits = {'lower': None, 'upper': None}
                    self.is_defining_annulus = False # Crucial: always reset this state
                    self.update_display()
                    self.update_measurements_display()
                    return # Stop further processing in this click event

                try:
                    # --- START: AUTO-DETECTION LOGIC ---
                    current_image_data = self.images[self.current_image_index]
                    raw_rgb_image = current_image_data['image'] # This is an RGB numpy.ndarray

                    self.image_processor.image = raw_rgb_image 
                    enhanced_image_for_detection = self.image_processor.enhance_image()

                    center_x = self.current_measurement['center'].x()
                    center_y = self.current_measurement['center'].y()
                    lower_rad = int(self.auto_detect_limits['lower'])
                    upper_rad = int(self.auto_detect_limits['upper'])

                    detected_radius_pixels = self.image_processor.detect_spectral_lines(
                        enhanced_image_for_detection, center_x, center_y, lower_rad, upper_rad
                    )

                    if detected_radius_pixels is not None:
                        self.current_measurement['radii'][ring_type_to_update] = detected_radius_pixels
                        self.current_measurement['type'] = ring_type_to_update 
                        QMessageBox.information(self, 'Success', 
                                                f"Auto-detection successful for {ring_type_to_update} ring. Radius: {detected_radius_pixels:.2f} pixels.")
                    else:
                        self.current_measurement['radii'][ring_type_to_update] = None # Ensure radius is cleared
                        QMessageBox.warning(self, 'Failure', 
                                            f"Auto-detection failed to find a clear {ring_type_to_update} ring within the specified limits.")
                    # --- END: AUTO-DETECTION LOGIC ---
                except Exception as e:
                    # Ensure radius is cleared in case of an error during processing
                    # Use the ring_type_to_update defined before the try block
                    if ring_type_to_update and ring_type_to_update in self.current_measurement['radii']:
                         self.current_measurement['radii'][ring_type_to_update] = None
                    QMessageBox.critical(self, "Processing Error", 
                                         f"An error occurred during {ring_type_to_update} ring detection: {str(e)}")
                finally:
                    # Reset state variables and update UI regardless of success or failure
                    self.current_mode = None 
                    self.auto_detect_limits = {'lower': None, 'upper': None}
                    self.is_defining_annulus = False # Crucial: always reset annulus definition state
                    self.update_display() # Update display to clear annulus lines and show new radius/lack thereof
                    self.update_measurements_display() # Update table if measurements changed

        elif self.current_mode and self.current_mode.startswith('auto_'):
            # This means it's the first click for annulus definition, show orange circle
            self.update_display()


        elif self.current_mode == 'calibrate':
            if len(self.calibration_points) < 2:
                self.calibration_points.append(pos)
                
                if len(self.calibration_points) == 2:
                    # Get calibration distance from user
                    distance, ok = QInputDialog.getDouble(
                        self,
                        'Enter Distance',
                        'Enter the distance between points (mm):',
                        value=10.0,
                        min=0.1,
                        max=1000.0,
                        decimals=2
                    )
                    
                    if ok:
                        # Calculate mm per pixel
                        dx = self.calibration_points[1].x() - self.calibration_points[0].x()
                        dy = self.calibration_points[1].y() - self.calibration_points[0].y()
                        pixel_distance = (dx * dx + dy * dy) ** 0.5
                        self.mm_per_pixel = distance / pixel_distance
                        self.images[self.current_image_index]['mm_per_pixel'] = self.mm_per_pixel
                        self.calibration_points = []
                        self.update_scale_display()
                    else:
                        self.calibration_points = []
                
                self.update_display()
        
        elif self.current_mode == 'center':
            # Set center point
            self.current_measurement['center'] = pos
            self.current_mode = None
            self.update_display()
            self.update_measurements_display()
        
        elif self.current_mode in ['inner', 'middle', 'outer']:
            # Measure radius
            if self.current_measurement['center'] is None:
                QMessageBox.warning(self, 'Warning', 'Please set center point first')
                return
            
            dx = pos.x() - self.current_measurement['center'].x()
            dy = pos.y() - self.current_measurement['center'].y()
            radius = (dx * dx + dy * dy) ** 0.5
            
            # Allow re-measuring any radius
            self.current_measurement['radii'][self.current_mode] = radius
            self.current_measurement['type'] = self.current_mode
            self.current_mode = None
            
            # Save measurement to current image
            self.images[self.current_image_index]['measurement'] = self.current_measurement
            
            self.update_display()
            self.update_measurements_display()
    
    def set_measurement_mode(self, mode):
        self.current_mode = mode
        if mode == 'center':
            self.initialize_measurement() # Resets radii for the new center
            self.is_defining_annulus = False # Ensure not in annulus mode
            self.auto_detect_limits = {'lower': None, 'upper': None}

        elif mode.startswith('auto_'):
            if self.current_measurement['center'] is None:
                QMessageBox.information(self, 'Set Center First', 
                                        'Please set the center point before defining an annulus for auto-detection.')
                self.current_mode = None
                self.is_defining_annulus = False
                return
            
            self.is_defining_annulus = True
            self.auto_detect_limits = {'lower': None, 'upper': None}
            # Optionally, update a status bar:
            # ring_type_display = mode.split('_')[1].capitalize()
            # self.statusBar().showMessage(f"Click to set lower annulus limit for {ring_type_display} ring.")
            print(f"Mode set to {mode}. Click to define annulus.") # Placeholder for status update
        else:
            # For manual modes 'inner', 'middle', 'outer' or 'calibrate'
            self.is_defining_annulus = False
            self.auto_detect_limits = {'lower': None, 'upper': None}
    
    def save_measurement(self):
        """Save current measurement with B-field value."""
        if not self.images or self.current_image_index < 0:
            QMessageBox.warning(self, 'Warning', 'No image loaded')
            return
            
        if self.current_measurement['center'] is None:
            QMessageBox.warning(self, 'Warning', 'Please set center point first')
            return
            
        if any(v is None for v in self.current_measurement['radii'].values()):
            QMessageBox.warning(self, 'Warning', 'Please measure all radii first')
            return
            
        # Get current and convert to magnetic field
        try:
            current = self.current_input.value()
            # Convert current to field using calibration
            slope, intercept = self.calibration_window.calibration_params
            field_gauss = slope * current + intercept
            magnetic_field = field_gauss / 1e4  # Convert Gauss to Tesla
        except (AttributeError, TypeError):
            QMessageBox.warning(self, 'Warning', 'Please calibrate the magnetic field first')
            return
        
        current_data = self.images[self.current_image_index]
        if not current_data.get('mm_per_pixel'):
            QMessageBox.warning(self, 'Warning', 'Please calibrate the image first')
            return
            
        # Create measurement object
        measurement = ZeemanMeasurement(
            B_field=magnetic_field,
            R_center=self.current_measurement['radii']['middle'] * current_data['mm_per_pixel'] if self.current_measurement['radii']['middle'] is not None else None,
            R_inner=self.current_measurement['radii']['inner'] * current_data['mm_per_pixel'] if self.current_measurement['radii']['inner'] is not None else None,
            R_outer=self.current_measurement['radii']['outer'] * current_data['mm_per_pixel'] if self.current_measurement['radii']['outer'] is not None else None,
            wavelength=self.wavelength_input.value() * 1e-9  # Convert nm to m
        )
        
        # Process measurement
        measurement = process_measurement(measurement)
        
        # Add to measurements list
        self.measurements.append(measurement)
        
        # Update table window
        self.table_window.update_table(self.measurements)
        
        # Clear current measurement
        self.current_measurement = {
            'center': None,
            'type': None,
            'radii': {'inner': None, 'middle': None, 'outer': None}
        }
        self.update_display()
        self.update_measurements_display()
    
    def show_plot(self):
        """Show the plot window."""
        self.plot_window.show()
        self.plot_window.raise_()
    
    def show_table(self):
        """Show the table window."""
        self.table_window.show()
        self.table_window.raise_()
    
    def show_results(self):
        """Show the results window."""
        self.results_window.show()
        self.results_window.raise_()
    
    def show_calibration(self):
        """Show the magnetic field calibration window."""
        self.calibration_window.show()
        self.calibration_window.raise_()
    
    def previous_image(self):
        """Switch to the previous image."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.initialize_measurement()
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def next_image(self):
        """Switch to the next image."""
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.initialize_measurement()
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Update control panel width
        if hasattr(self, 'control_scroll'):
            control_width = int(self.width() * 0.25)
            self.control_scroll.setFixedWidth(control_width)
        # Update display if there's an image
        self.update_display()
    
    def update_navigation(self):
        """Update the navigation buttons and image label."""
        self.prev_image_btn.setEnabled(self.current_image_index > 0)
        self.next_image_btn.setEnabled(self.current_image_index < len(self.images) - 1)
        
        if self.images and self.current_image_index >= 0:
            self.image_label.setText(f"Image {self.current_image_index + 1} of {len(self.images)}")
        else:
            self.image_label.setText("No image loaded")
    
    def initialize_measurement(self):
        """Initialize or reset the current measurement."""
        self.current_measurement = {
            'center': None,
            'type': None,
            'radii': {'inner': None, 'middle': None, 'outer': None}
        }
    
    def zoom_in(self):
        """Zoom in on the image."""
        self.scale_factor *= 1.2
        self.update_display()
    
    def zoom_out(self):
        """Zoom out from the image."""
        self.scale_factor /= 1.2
        self.update_display()
    
    def reset_view(self):
        """Reset the zoom level."""
        self.scale_factor = 1.0
        self.update_display()
    
    def update_scale_display(self):
        """Update the scale/calibration display."""
        if not self.images or self.current_image_index < 0:
            self.calibration_label.setText("Scale: Not calibrated")
            return
                
        img_data = self.images[self.current_image_index]
        if img_data.get('mm_per_pixel') is not None:
            self.calibration_label.setText(f"Scale: {img_data['mm_per_pixel']:.4f} mm/pixel")
        else:
            self.calibration_label.setText("Scale: Not calibrated")
    
    def update_measurements_display(self):
        """Update the measurements table."""
        self.measurements_table.setRowCount(len(self.measurements))
        
        for i, measurement in enumerate(self.measurements):
            # Get current from B field using calibration
            try:
                slope, intercept = self.calibration_window.calibration_params
                current = (measurement.B_field * 1e4 - intercept) / slope
            except (AttributeError, TypeError):
                current = 0
            
            # Current
            current_item = QTableWidgetItem(f"{current:.3f}")
            current_item.setFlags(current_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.measurements_table.setItem(i, 0, current_item)
            
            # B Field
            field_item = QTableWidgetItem(f"{measurement.B_field:.6f}")
            field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.measurements_table.setItem(i, 1, field_item)
            
            # Create delete button
            delete_btn = QPushButton('Delete')
            delete_btn.clicked.connect(lambda checked, row=i: self.delete_measurement(row))
            self.measurements_table.setCellWidget(i, 2, delete_btn)
        
        self.measurements_table.resizeColumnsToContents()
        
    def delete_measurement(self, index):
        """Delete a measurement and update displays."""
        if 0 <= index < len(self.measurements):
            # Remove the measurement
            self.measurements.pop(index)
            
            # Update displays
            self.update_measurements_display()
            self.table_window.update_table(self.measurements)
            
            # Update plot if measurements exist
            if self.measurements:
                self.plot_window.plot_data(self.measurements)
            
            QMessageBox.information(self, 'Success', f'Measurement {index + 1} deleted')
    
    def calculate_results(self):
        """Calculate final results and update all windows."""
        if not self.measurements:
            QMessageBox.warning(self, 'Warning', 'No measurements available')
            return
        
        # Calculate Bohr magneton and specific charge
        results = calculate_bohr_magneton(self.measurements)
        
        # Update windows
        self.plot_window.plot_data(self.measurements)
        self.table_window.update_table(self.measurements)
        self.results_window.update_results(results)
        
        # Show all windows
        self.show_plot()
        self.show_table()
        self.show_results()
    
    def export_to_csv(self):
        """Export measurements to a CSV file."""
        if not self.measurements:
            QMessageBox.warning(self, 'Warning', 'No measurements to export')
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Measurements',
            '',
            'CSV Files (*.csv)'
        )
        
        if not file_path:
            return

        # Constants
        L = 150  # Distance in mm
        n = 1.46  # Refractive index
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Write header
            writer.writerow([
                'I(A)', 'B(G)', 
                'R_i(mm)', 'R_c(mm)', 'R_o(mm)',
                'α_i(deg)', 'α_c(deg)', 'α_o(deg)',
                'β_i(deg)', 'β_c(deg)', 'β_o(deg)',
                'Δλ_i(nm)', 'Δλ_o(nm)',
                'ΔE_i(eV)', 'ΔE_o(eV)'
            ])
            
            # Write data
            for m in self.measurements:
                # Get magnetic field in Gauss
                B_field = m.B_field * 1e4  # Convert T to G
                
                # Get current from magnetic field using calibration
                try:
                    slope, intercept = self.calibration_window.calibration_params
                    current = (B_field - intercept) / slope  # Inverse of B = slope * I + intercept
                except (AttributeError, TypeError):
                    QMessageBox.warning(self, 'Warning', 'Please calibrate the magnetic field first')
                    return
                
                # Calculate angles
                def calc_angles(r_mm):
                    if r_mm is None:
                        return None, None
                    alpha = np.arctan(r_mm / L)
                    beta = np.arcsin(np.sin(alpha) / n)
                    return alpha, beta
                
                # Calculate angles for each radius
                alpha_i, beta_i = calc_angles(m.R_inner)
                alpha_c, beta_c = calc_angles(m.R_center)
                alpha_o, beta_o = calc_angles(m.R_outer)
                
                # Format values with appropriate precision
                def format_val(val, precision=6):
                    return f"{val:.{precision}f}" if val is not None else ""
                
                row = [
                    format_val(current, 2),  # Convert to A assuming 10000 G/A
                    format_val(B_field, 2),  # Gauss
                    format_val(m.R_inner, 3) if m.R_inner else "",
                    format_val(m.R_center, 3) if m.R_center else "",
                    format_val(m.R_outer, 3) if m.R_outer else "",
                    format_val(np.degrees(alpha_i), 4) if alpha_i is not None else "",
                    format_val(np.degrees(alpha_c), 4) if alpha_c is not None else "",
                    format_val(np.degrees(alpha_o), 4) if alpha_o is not None else "",
                    format_val(np.degrees(beta_i), 4) if beta_i is not None else "",
                    format_val(np.degrees(beta_c), 4) if beta_c is not None else "",
                    format_val(np.degrees(beta_o), 4) if beta_o is not None else "",
                    format_val(m.delta_lambda_i * 1e9, 3) if m.delta_lambda_i else "",  # nm
                    format_val(m.delta_lambda_o * 1e9, 3) if m.delta_lambda_o else "",  # nm
                    format_val(m.delta_E_i / 1.602176634e-19, 6) if m.delta_E_i else "",  # eV
                    format_val(m.delta_E_o / 1.602176634e-19, 6) if m.delta_E_o else ""   # eV
                ]
                writer.writerow(row)
                
            QMessageBox.information(self, 'Success', f'Measurements exported to {file_path}')

    def fill_test_data(self):
        """Fill test data for quick testing."""
        # Test data for magnetic field calibration
        calibration_points = [
            (0.0, 0),
            (0.5, 5000),
            (1.0, 10000),
            (1.5, 15000),
            (2.0, 20000)
        ]
        
        # Show calibration window
        self.calibration_window.show()
        
        # Fill calibration data
        for current, field in calibration_points:
            self.calibration_window.current_input.setValue(current)
            self.calibration_window.field_input.setValue(field)
            self.calibration_window.add_point()
        
        # Calculate calibration parameters
        currents = [point[0] for point in calibration_points]
        fields = [point[1] for point in calibration_points]
        coeffs = np.polyfit(currents, fields, 1)
        self.calibration_window.calibration_params = (coeffs[0], coeffs[1])  # slope, intercept
        
        # Update calibration display
        slope, intercept = self.calibration_window.calibration_params
        self.calibration_window.status_label.setText(
            f'Calibration: B = {slope:.1f} * I + {intercept:.1f} (Gauss)'
        )
        
        # Update plot
        self.calibration_window.update_plot()
        
        # Create a test image (black background with white dots)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        center_x, center_y = 320, 240
        
        # Add spectral lines for each test case
        test_cases = [
            {'current': 0.5, 'radii': (50, 60, 70)},  # Inner, Center, Outer radii in pixels
            {'current': 1.0, 'radii': (55, 65, 75)},
            {'current': 1.5, 'radii': (60, 70, 80)},
            {'current': 2.0, 'radii': (65, 75, 85)}
        ]
        
        for case in test_cases:
            # Create image with spectral lines
            test_img = img.copy()
            for radius in case['radii']:
                cv2.circle(test_img, (center_x, center_y), radius, (255, 255, 255), 2)
            
            # Add image to the list
            self.images.append({
                'image': test_img,
                'mm_per_pixel': 0.1  # 0.1 mm per pixel for testing
            })
        
        # Set first image as current
        self.current_image_index = 0
        self.update_display()
        self.update_navigation()
        
        # Take measurements for each test case
        for i, case in enumerate(test_cases):
            # Set center point
            self.current_measurement = {
                'center': QPoint(center_x, center_y),
                'type': None,
                'radii': {'inner': None, 'middle': None, 'outer': None}
            }
            
            # Set radii
            self.current_measurement['radii']['inner'] = case['radii'][0]
            self.current_measurement['radii']['middle'] = case['radii'][1]
            self.current_measurement['radii']['outer'] = case['radii'][2]
            
            # Set current value
            self.current_input.setValue(case['current'])
            
            # Save measurement
            self.save_measurement()
            
            # Move to next image if not last
            if i < len(test_cases) - 1:
                self.next_image()
        
        QMessageBox.information(self, 'Success', 'Test data has been loaded. Press Ctrl+S to save measurements.')
