from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QFileDialog, QScrollArea,
                               QGroupBox, QMessageBox, QInputDialog, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import csv

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Zeeman Effect Analysis')
        self.setGeometry(100, 100, 1200, 800)
        
        # Measurement variables
        self.images = []  # List of loaded images with their measurements
        self.current_image_index = -1
        self.current_measurement = None
        self.current_mode = None
        self.scale_factor = 1.0
        
        # Initialize measurement
        self.initialize_measurement()
        
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
        image_scroll.setMinimumSize(800, 600)
        
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.mousePressEvent = self.image_clicked
        image_container_layout.addWidget(self.image_display)
        image_container_layout.addStretch()
        
        image_scroll.setWidget(image_container)
        content_layout.addWidget(image_scroll, 3)  # 3:1 ratio
        
        # Create control panel with scroll area
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(300)
        control_scroll.setMaximumWidth(400)
        
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
        
        reset_btn = QPushButton("Reset Measurements")
        reset_btn.clicked.connect(self.reset_measurements)
        measurement_layout.addWidget(reset_btn)
        
        measurement_group.setLayout(measurement_layout)
        control_layout.addWidget(measurement_group)
        
        # Measurements display
        measurements_group = QGroupBox("Measurements")
        measurements_layout = QVBoxLayout(measurements_group)
        
        self.measurements_label = QLabel("No measurements")
        self.measurements_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.measurements_label.setWordWrap(True)
        measurements_scroll = QScrollArea()
        measurements_scroll.setWidget(self.measurements_label)
        measurements_scroll.setWidgetResizable(True)
        measurements_scroll.setMinimumHeight(200)
        measurements_layout.addWidget(measurements_scroll)
        
        measurements_group.setLayout(measurements_layout)
        control_layout.addWidget(measurements_group)
        
        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        # Add input fields for wavelength and magnetic field
        param_layout = QHBoxLayout()
        
        wavelength_layout = QVBoxLayout()
        wavelength_label = QLabel("Wavelength (nm):")
        self.wavelength_input = QDoubleSpinBox()
        self.wavelength_input.setRange(0, 1000)
        self.wavelength_input.setValue(546.1)  # Default wavelength for mercury green line
        self.wavelength_input.setDecimals(1)
        wavelength_layout.addWidget(wavelength_label)
        wavelength_layout.addWidget(self.wavelength_input)
        param_layout.addLayout(wavelength_layout)
        
        field_layout = QVBoxLayout()
        field_label = QLabel("Magnetic Field (T):")
        self.field_input = QDoubleSpinBox()
        self.field_input.setRange(0, 10)
        self.field_input.setValue(1.0)  # Default field strength
        self.field_input.setDecimals(3)
        field_layout.addWidget(field_label)
        field_layout.addWidget(self.field_input)
        param_layout.addLayout(field_layout)
        
        results_layout.addLayout(param_layout)
        
        calculate_btn = QPushButton("Calculate Results")
        calculate_btn.clicked.connect(self.calculate_results)
        results_layout.addWidget(calculate_btn)
        
        export_btn = QPushButton("Export Measurements")
        export_btn.clicked.connect(self.export_measurements)
        results_layout.addWidget(export_btn)
        
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setMinimumHeight(200)
        results_scroll.setWidget(self.results_label)
        results_layout.addWidget(results_scroll)
        
        results_group.setLayout(results_layout)
        # Add control panel to content layout
        control_layout.addWidget(results_group)
        control_layout.addStretch()
        
        control_scroll.setWidget(control_panel)
        content_layout.addWidget(control_scroll)
        
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
            cv2.circle(display_img, (center.x(), center.y()), 3, (255, 0, 0), -1)

            # Draw radii with different colors
            colors = {
                'inner': (255, 0, 0),   # Blue
                'middle': (0, 255, 0),  # Green
                'outer': (0, 0, 255)    # Red
            }
            
            for radius_type, radius in self.current_measurement['radii'].items():
                if radius is not None:
                    color = colors.get(radius_type, (0, 0, 255))
                    cv2.circle(display_img, (center.x(), center.y()), int(radius), color, 1)

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
            self.images[self.current_image_index]['measurement'] = None
            self.initialize_measurement()
            self.update_display()
            self.update_measurements_display()
    
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

        # Convert screen coordinates to image coordinates
        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return

        if self.current_mode == 'calibrate':
            # Handle calibration point selection
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
        self.update_display()
    
    def set_measurement_mode(self, mode):
        self.current_mode = mode
        if mode == 'center':
            self.initialize_measurement()
    
    def calculate_scale(self):
        if not self.images or self.current_image_index < 0:
            return

        current_data = self.images[self.current_image_index]
        if len(current_data['calibration_points']) != 2:
            return

        # Calculate distance in pixels
        dx = current_data['calibration_points'][1][0] - current_data['calibration_points'][0][0]
        dy = current_data['calibration_points'][1][1] - current_data['calibration_points'][0][1]
        pixel_distance = np.sqrt(dx**2 + dy**2)

        # Get real distance in mm from user
        real_distance, ok = QInputDialog.getDouble(
            self,
            'Enter Real Distance',
            'Enter the real distance in millimeters:',
            1.0, 0.0, 1000.0, 2
        )

        if ok and pixel_distance > 0:
            current_data['mm_per_pixel'] = real_distance / pixel_distance
            self.update_scale_display()
    
    def initialize_measurement(self):
        """Initialize or reset the current measurement and calibration."""
        self.current_measurement = {
            'center': None,
            'points': [],
            'radii': {'inner': None, 'middle': None, 'outer': None},
            'type': None  # 'inner', 'middle', or 'outer'
        }
        
        # Initialize calibration variables
        self.calibration_points = []
        self.mm_per_pixel = None
        self.calibration_distance_mm = 10.0  # Default calibration distance
    
    def update_scale_display(self):
        if not self.images or self.current_image_index < 0:
            self.calibration_label.setText("Scale: Not calibrated")
            return
                
        img_data = self.images[self.current_image_index]
        if img_data.get('mm_per_pixel') is not None:
            self.calibration_label.setText(f"Scale: {img_data['mm_per_pixel']:.4f} mm/pixel")
        else:
            self.calibration_label.setText("Scale: Not calibrated")
    
    def update_measurements_display(self):
        if not self.images or self.current_image_index < 0:
            self.measurements_label.setText("No measurements")
            return

        img_data = self.images[self.current_image_index]
        text = f"Image {self.current_image_index + 1} of {len(self.images)}\n\n"

        # Show calibration status
        if img_data.get('mm_per_pixel') is not None:
            text += f"Calibration: {img_data['mm_per_pixel']:.4f} mm/pixel\n\n"
            self.calibration_label.setText(f"Scale: {img_data['mm_per_pixel']:.4f} mm/pixel")
        else:
            text += "Not calibrated\n\n"
            self.calibration_label.setText("Scale: Not calibrated")

        # Current measurement
        if self.current_measurement['center'] is not None:
            text += "Current measurement:\n"
            text += "Center point set\n"
            for radius_type, radius in self.current_measurement['radii'].items():
                if radius is not None:
                    radius_mm = radius * (img_data.get('mm_per_pixel', 0) or 0)
                    text += f"{radius_type.capitalize()} radius: {radius:.1f} px"
                    if img_data.get('mm_per_pixel'):
                        text += f" ({radius_mm:.2f} mm)"
                    text += "\n"
                else:
                    text += f"{radius_type.capitalize()} radius: Not measured\n"

        self.measurements_label.setText(text)
    
    def zoom_in(self):
        self.scale_factor *= 1.2
        self.update_display()
    
    def zoom_out(self):
        self.scale_factor /= 1.2
        self.update_display()
    
    def reset_view(self):
        self.scale_factor = 1.0
        self.update_display()
        
    def update_navigation(self):
        """Update the navigation buttons and image label."""
        self.prev_image_btn.setEnabled(self.current_image_index > 0)
        self.next_image_btn.setEnabled(self.current_image_index < len(self.images) - 1)
        
        if self.current_image_index >= 0:
            self.image_label.setText(f"Image {self.current_image_index + 1} of {len(self.images)}")
        else:
            self.image_label.setText("No image loaded")
            
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
    
    def calculate_results(self):
        if not self.images:
            QMessageBox.warning(self, 'Warning', 'No images loaded')
            return

        wavelength = self.wavelength_input.value()  # nm
        magnetic_field = self.field_input.value()  # Tesla

        # Collect all valid measurements
        all_measurements = []
        for img_data in self.images:
            measurement = img_data.get('measurement')
            if not measurement:
                continue
                
            radii = measurement.get('radii', {})
            if not all(radii.get(k) for k in ['inner', 'middle', 'outer']):
                continue
                
            all_measurements.append(radii)

        if not all_measurements:
            QMessageBox.warning(self, 'Warning', 'No complete measurement sets found')
            return

        # Calculate results for each complete set
        results = []
        for i, m in enumerate(all_measurements, 1):
            # Calculate wavelength shifts
            pixel_to_wavelength = wavelength / m['middle']
            inner_shift = abs(m['inner'] - m['middle']) * pixel_to_wavelength
            outer_shift = abs(m['outer'] - m['middle']) * pixel_to_wavelength

            # Calculate Bohr magneton
            bohr_magneton = (inner_shift + outer_shift) * 1e-9 / (4 * magnetic_field)

            # Calculate specific charge
            specific_charge = bohr_magneton / 9.274e-24  # Divide by standard Bohr magneton

            results.append({
                'set': i,
                'inner_shift': inner_shift,
                'outer_shift': outer_shift,
                'bohr_magneton': bohr_magneton,
                'specific_charge': specific_charge
            })

        # Calculate averages
        avg_bohr_magneton = np.mean([r['bohr_magneton'] for r in results])
        avg_specific_charge = np.mean([r['specific_charge'] for r in results])
        std_bohr_magneton = np.std([r['bohr_magneton'] for r in results])
        std_specific_charge = np.std([r['specific_charge'] for r in results])
        
        # Display results
        text = "Results:\n\n"
        for r in results:
            text += f"Set {r['set']}:\n"
            text += f"Inner shift: {r['inner_shift']:.2f} nm\n"
            text += f"Outer shift: {r['outer_shift']:.2f} nm\n"
            text += f"Bohr magneton: {r['bohr_magneton']:.3e} J/T\n"
            text += f"Specific charge: {r['specific_charge']:.3e} C/kg\n\n"
        
        text += "Average values:\n"
        text += f"Bohr magneton: {avg_bohr_magneton:.3e} ± {std_bohr_magneton:.3e} J/T\n"
        text += f"Specific charge: {avg_specific_charge:.3e} ± {std_specific_charge:.3e} C/kg"
        
        self.results_label.setText(text)

    def export_measurements(self):
        """Export measurements to a CSV file."""
        if not self.images:
            QMessageBox.warning(self, 'Warning', 'No measurements to export')
            return
            
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Measurements',
            '',
            'CSV Files (*.csv)'
        )
        
        if not file_path:
            return
            
        magnetic_field = self.field_input.value()
        wavelength = self.wavelength_input.value()
        

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Set', 'Inner Radius (px)', 'Middle Radius (px)', 'Outer Radius (px)',
                           'Inner Radius (mm)', 'Middle Radius (mm)', 'Outer Radius (mm)',
                           'Bohr Magneton (J/T)', 'Specific Charge (C/kg)'])
            
            # Export measurements from each image
            for i, image in enumerate(self.images, 1):
                if 'measurement' not in image or not image['measurement']:
                    continue
                    
                m = image['measurement']
                if not m['radii']['middle']:
                    continue
                    
                # Calculate physical quantities
                pixel_to_wavelength = wavelength * 1e-9 / m['radii']['middle']
                delta_lambda_avg = (
                    abs(m['radii']['inner'] - m['radii']['middle']) +
                    abs(m['radii']['outer'] - m['radii']['middle'])
                ) * pixel_to_wavelength / 2
                
                bohr_magneton = delta_lambda_avg / (4 * magnetic_field)
                specific_charge = bohr_magneton / 9.274e-24  # Divide by standard Bohr magneton
                
                # Convert radii to mm
                radii_mm = {k: v * image.get('mm_per_pixel', 0) if v is not None else 0
                           for k, v in m['radii'].items()}
                
                writer.writerow([
                    i,
                    m['radii']['inner'] or 0,
                    m['radii']['middle'] or 0,
                    m['radii']['outer'] or 0,
                    radii_mm['inner'],
                    radii_mm['middle'],
                    radii_mm['outer'],
                    bohr_magneton,
                    specific_charge
                ])
