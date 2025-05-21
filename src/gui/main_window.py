import csv
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea,
    QGroupBox, QMessageBox, QInputDialog, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QApplication
)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QScreen
from typing import Optional 
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
from src.gui.image_display_manager import ImageDisplayManager
from src.gui.measurement_controller import MeasurementController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle('Zeeman Effect Analysis')
        
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        
        window_width = int(self.screen_width * 0.8)
        window_height = int(self.screen_height * 0.8)
        
        x = (self.screen_width - window_width) // 2
        y = (self.screen_height - window_height) // 2
        
        self.setGeometry(x, y, window_width, window_height)
        
        self.shortcut_test = QShortcut(QKeySequence('Ctrl+T'), self)
        self.shortcut_test.activated.connect(self.fill_test_data)
        
        self.images = []  # List of loaded images with their measurements
        self.current_image_index = -1
        self.current_measurement = None
        self.current_mode = None
        
        self.image_processor = ImageProcessor()

        self.current_mode = None  # 'calibrate', 'center', 'inner', 'middle', 'outer', 'auto_inner', 'auto_middle', 'auto_outer'
        self.calibration_points = []
        self.current_measurement = {
            'center': None,
            'type': None,
            'radii': {'inner': None, 'middle': None, 'outer': None}
        }
        self.mm_per_pixel = None
        self.calibration_distance_mm = 10.0  # Default calibration distance

        self.auto_detect_limits = {'lower': None, 'upper': None}
        self.is_defining_annulus = False
        
        self.measurements = []  # List of ZeemanMeasurement objects
        
        self.calibration_window = CalibrationWindow()
        
        self.create_ui() # self.image_display is created in here

        self.measurement_controller = MeasurementController(self)

        self.image_display_manager = ImageDisplayManager(self.image_display, self)

    def _create_image_controls_group(self) -> QGroupBox:
        image_group = QGroupBox('Image Controls')
        image_layout = QVBoxLayout()

        load_btn = QPushButton('Load Image')
        load_btn.clicked.connect(self.load_image)
        image_layout.addWidget(load_btn)

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
        return image_group

    def _create_calibration_group(self) -> QGroupBox:
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        
        calibrate_btn = QPushButton("Calibrate Scale")
        calibrate_btn.clicked.connect(lambda: self.set_measurement_mode('calibrate'))
        calibration_layout.addWidget(calibrate_btn)
        
        self.calibration_label = QLabel("Scale: Not calibrated") # self.calibration_label is an attribute
        calibration_layout.addWidget(self.calibration_label)
        
        return calibration_group

    def _create_measurement_controls_group(self) -> QGroupBox:
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
        
        return measurement_group

    def _create_measurements_display_group(self) -> QGroupBox:
        measurements_group = QGroupBox("Measurements")
        measurements_layout = QVBoxLayout(measurements_group)
        
        self.measurements_table = QTableWidget() # self.measurements_table is an attribute
        self.measurements_table.setColumnCount(3)
        self.measurements_table.setHorizontalHeaderLabels(['Current (A)', 'B Field (T)', 'Actions'])
        self.measurements_table.horizontalHeader().setStretchLastSection(True)
        measurements_layout.addWidget(self.measurements_table)
        
        return measurements_group

    def _create_results_group(self) -> QGroupBox:
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        params_layout = QHBoxLayout()
        
        current_group = QGroupBox('Magnetic Field')
        current_group_layout = QVBoxLayout()
        
        current_layout = QHBoxLayout()
        current_label = QLabel('Current (A):')
        self.current_input = QDoubleSpinBox() # self.current_input is an attribute
        self.current_input.setRange(0, 100)
        self.current_input.setDecimals(3)
        self.current_input.setSingleStep(0.1)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_input)
        current_group_layout.addLayout(current_layout)
        
        self.calibrate_btn = QPushButton('Calibrate Field') # self.calibrate_btn is an attribute
        self.calibrate_btn.clicked.connect(self.show_calibration)
        current_group_layout.addWidget(self.calibrate_btn)
        
        current_group.setLayout(current_group_layout)
        params_layout.addWidget(current_group)
        
        # Wavelength input
        wavelength_layout = QVBoxLayout()
        wavelength_label = QLabel('Wavelength (nm):')
        self.wavelength_input = QDoubleSpinBox() # self.wavelength_input is an attribute
        self.wavelength_input.setRange(300, 1000)
        self.wavelength_input.setDecimals(1)
        self.wavelength_input.setValue(643.8)  # Default wavelength
        wavelength_layout.addWidget(wavelength_label)
        wavelength_layout.addWidget(self.wavelength_input)
        params_layout.addLayout(wavelength_layout)
        
        results_layout.addLayout(params_layout)
        
        self.save_measurement_btn = QPushButton('Save Measurement') # self.save_measurement_btn is an attribute
        self.save_measurement_btn.clicked.connect(self.save_measurement)
        results_layout.addWidget(self.save_measurement_btn)
        
        buttons_layout = QHBoxLayout()
        
        self.calculate_btn = QPushButton('Calculate Results') # self.calculate_btn is an attribute
        self.calculate_btn.clicked.connect(self.calculate_results)
        buttons_layout.addWidget(self.calculate_btn)
        
        self.show_plot_btn = QPushButton('Show Plot') # self.show_plot_btn is an attribute
        self.show_plot_btn.clicked.connect(self.show_plot)
        buttons_layout.addWidget(self.show_plot_btn)
        
        self.show_table_btn = QPushButton('Show Data Table') # self.show_table_btn is an attribute
        self.show_table_btn.clicked.connect(self.show_table)
        buttons_layout.addWidget(self.show_table_btn)
        
        self.show_results_btn = QPushButton('Show Results') # self.show_results_btn is an attribute
        self.show_results_btn.clicked.connect(self.show_results)
        buttons_layout.addWidget(self.show_results_btn)
        
        results_layout.addLayout(buttons_layout)
        
        self.export_btn = QPushButton('Export to CSV') # self.export_btn is an attribute
        self.export_btn.clicked.connect(self.export_to_csv)
        results_layout.addWidget(self.export_btn)
        
        return results_group

    def create_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
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
        
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
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
        
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        control_width = int(self.width() * 0.4)
        self.control_scroll.setFixedWidth(control_width)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        control_layout.addWidget(self._create_image_controls_group())
        control_layout.addWidget(self._create_calibration_group())
        control_layout.addWidget(self._create_measurement_controls_group())
        control_layout.addWidget(self._create_measurements_display_group())
        control_layout.addWidget(self._create_results_group())
        
        self.plot_window = PlotWindow()
        self.table_window = TableWindow()
        self.results_window = ResultsWindow()

        control_layout.addStretch()
        
        self.control_scroll.setWidget(control_panel)
        content_layout.addWidget(self.control_scroll, 25)  # 25% of width
        content_layout.setStretch(1, 25)  # Control panel stretch factor
        
        self.setCentralWidget(central_widget)
        
        self.update_navigation()
    
    def update_display(self):
        self.image_display_manager.redraw_image_with_overlays()
    
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
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.images.append({
                'image': image,
                'calibration_points': [],
                'mm_per_pixel': None,
                'measurement': None
            })
            
            self.current_image_index = len(self.images) - 1
            self.initialize_measurement()
            if hasattr(self, 'image_display_manager'): # Ensure manager exists
                self.image_display_manager.scale_factor = 1.0 # Reset zoom
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def reset_measurements(self):
        if self.current_image_index >= 0:
            self.measurement_controller.reset_all_measurement_states()
            
            self.current_measurement = self.measurement_controller.current_measurement
            self.current_mode = self.measurement_controller.current_mode
            self.is_defining_annulus = self.measurement_controller.is_defining_annulus
            self.auto_detect_limits = self.measurement_controller.auto_detect_limits
            
            self.update_display()
            self.update_measurements_display()
    
    def get_image_coordinates(self, event_pos: QPoint) -> Optional[QPoint]:
        return self.image_display_manager.get_image_coordinates(event_pos)
    
    def image_clicked(self, event):
        if not self.images or self.current_image_index < 0:
            return

        pos = self.get_image_coordinates(event.pos())
        if pos is None:
            return
            
        self.measurement_controller.handle_image_click(pos)
        
        self.current_measurement = self.measurement_controller.current_measurement
        
        if self.current_image_index >= 0 and self.current_measurement:
            self.images[self.current_image_index]['measurement'] = self.current_measurement
    
    def set_measurement_mode(self, mode):
        self.measurement_controller.set_mode(mode)
        
        self.current_mode = self.measurement_controller.current_mode
        self.is_defining_annulus = self.measurement_controller.is_defining_annulus
        self.auto_detect_limits = self.measurement_controller.auto_detect_limits
        
        self.update_display()
    
    def save_measurement(self):
        if not self.images or self.current_image_index < 0:
            QMessageBox.warning(self, 'Warning', 'No image loaded')
            return
            
        if self.current_measurement['center'] is None:
            QMessageBox.warning(self, 'Warning', 'Please set center point first')
            return
            
        if any(v is None for v in self.current_measurement['radii'].values()):
            QMessageBox.warning(self, 'Warning', 'Please measure all radii first')
            return
            
        try:
            current = self.current_input.value()
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
            
        measurement = ZeemanMeasurement(
            B_field=magnetic_field,
            R_center=self.current_measurement['radii']['middle'] * current_data['mm_per_pixel'] if self.current_measurement['radii']['middle'] is not None else None,
            R_inner=self.current_measurement['radii']['inner'] * current_data['mm_per_pixel'] if self.current_measurement['radii']['inner'] is not None else None,
            R_outer=self.current_measurement['radii']['outer'] * current_data['mm_per_pixel'] if self.current_measurement['radii']['outer'] is not None else None,
            wavelength=self.wavelength_input.value() * 1e-9  # Convert nm to m
        )
        
        measurement = process_measurement(measurement)
        
        self.measurements.append(measurement)
        
        self.table_window.update_table(self.measurements)
        
        self.current_measurement = {
            'center': None,
            'type': None,
            'radii': {'inner': None, 'middle': None, 'outer': None}
        }
        self.update_display()
        self.update_measurements_display()
    
    def show_plot(self):
        self.plot_window.show()
        self.plot_window.raise_()
    
    def show_table(self):
        self.table_window.show()
        self.table_window.raise_()
    
    def show_results(self):
        self.results_window.show()
        self.results_window.raise_()
    
    def show_calibration(self):
        self.calibration_window.show()
        self.calibration_window.raise_()
    
    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.initialize_measurement()
            if hasattr(self, 'image_display_manager'):
                self.image_display_manager.scale_factor = 1.0 # Reset zoom
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.initialize_measurement()
            if hasattr(self, 'image_display_manager'):
                self.image_display_manager.scale_factor = 1.0 # Reset zoom
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'control_scroll'):
            control_width = int(self.width() * 0.25)
            self.control_scroll.setFixedWidth(control_width)
        self.update_display()
    
    def update_navigation(self):
        self.prev_image_btn.setEnabled(self.current_image_index > 0)
        self.next_image_btn.setEnabled(self.current_image_index < len(self.images) - 1)
        
        if self.images and self.current_image_index >= 0:
            self.image_label.setText(f"Image {self.current_image_index + 1} of {len(self.images)}")
        else:
            self.image_label.setText("No image loaded")
    
    def initialize_measurement(self):
        self.measurement_controller.initialize_for_new_measurement()
        
        self.current_measurement = self.measurement_controller.current_measurement
    
    def zoom_in(self):
        self.image_display_manager.zoom_in()
    
    def zoom_out(self):
        self.image_display_manager.zoom_out()
    
    def reset_view(self):
        self.image_display_manager.reset_view()
    
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
        self.measurements_table.setRowCount(len(self.measurements))
        
        for i, measurement in enumerate(self.measurements):
            try:
                slope, intercept = self.calibration_window.calibration_params
                current = (measurement.B_field * 1e4 - intercept) / slope
            except (AttributeError, TypeError):
                current = 0
            
            current_item = QTableWidgetItem(f"{current:.3f}")
            current_item.setFlags(current_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.measurements_table.setItem(i, 0, current_item)
            
            field_item = QTableWidgetItem(f"{measurement.B_field:.6f}")
            field_item.setFlags(field_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.measurements_table.setItem(i, 1, field_item)
            
            delete_btn = QPushButton('Delete')
            delete_btn.clicked.connect(lambda checked, row=i: self.delete_measurement(row))
            self.measurements_table.setCellWidget(i, 2, delete_btn)
        
        self.measurements_table.resizeColumnsToContents()
        
    def delete_measurement(self, index):
        if 0 <= index < len(self.measurements):
            self.measurements.pop(index)
            
            self.update_measurements_display()
            self.table_window.update_table(self.measurements)
            
            if self.measurements:
                self.plot_window.plot_data(self.measurements)
            
            QMessageBox.information(self, 'Success', f'Measurement {index + 1} deleted')
    
    def calculate_results(self):
        if not self.measurements:
            QMessageBox.warning(self, 'Warning', 'No measurements available')
            return
        
        results = calculate_bohr_magneton(self.measurements)
        
        self.plot_window.plot_data(self.measurements)
        self.table_window.update_table(self.measurements)
        self.results_window.update_results(results)
        
        self.show_plot()
        self.show_table()
        self.show_results()
    
    def export_to_csv(self):
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

        L = 150  # Distance in mm
        n = 1.46  # Refractive index
        
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            writer.writerow([
                'I(A)', 'B(G)', 
                'R_i(mm)', 'R_c(mm)', 'R_o(mm)',
                'α_i(deg)', 'α_c(deg)', 'α_o(deg)',
                'β_i(deg)', 'β_c(deg)', 'β_o(deg)',
                'Δλ_i(nm)', 'Δλ_o(nm)',
                'ΔE_i(eV)', 'ΔE_o(eV)'
            ])
            
            for m in self.measurements:
                B_field = m.B_field * 1e4  # Convert T to G
                
                try:
                    slope, intercept = self.calibration_window.calibration_params
                    current = (B_field - intercept) / slope  # Inverse of B = slope * I + intercept
                except (AttributeError, TypeError):
                    QMessageBox.warning(self, 'Warning', 'Please calibrate the magnetic field first')
                    return
                
                def calc_angles(r_mm):
                    if r_mm is None:
                        return None, None
                    alpha = np.arctan(r_mm / L)
                    beta = np.arcsin(np.sin(alpha) / n)
                    return alpha, beta
                
                alpha_i, beta_i = calc_angles(m.R_inner)
                alpha_c, beta_c = calc_angles(m.R_center)
                alpha_o, beta_o = calc_angles(m.R_outer)
                
                def format_val(val, precision=6):
                    return f"{val:.{precision}f}" if val is not None else ""
                
                row = [
                    format_val(current, 2),  
                    format_val(B_field, 2),  
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
        calibration_points = [
            (0.0, 0),
            (0.5, 5000),
            (1.0, 10000),
            (1.5, 15000),
            (2.0, 20000)
        ]
        
        self.calibration_window.show()
        
        for current, field in calibration_points:
            self.calibration_window.current_input.setValue(current)
            self.calibration_window.field_input.setValue(field)
            self.calibration_window.add_point()
        
        currents = [point[0] for point in calibration_points]
        fields = [point[1] for point in calibration_points]
        coeffs = np.polyfit(currents, fields, 1)
        self.calibration_window.calibration_params = (coeffs[0], coeffs[1])
        
        slope, intercept = self.calibration_window.calibration_params
        self.calibration_window.status_label.setText(
            f'Calibration: B = {slope:.1f} * I + {intercept:.1f} (Gauss)'
        )
        
        self.calibration_window.update_plot()
        
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        center_x, center_y = 320, 240
        
        test_cases = [
            {'current': 0.5, 'radii': (50, 60, 70)},
            {'current': 1.0, 'radii': (55, 65, 75)},
            {'current': 1.5, 'radii': (60, 70, 80)},
            {'current': 2.0, 'radii': (65, 75, 85)}
        ]
        
        for case in test_cases:
            test_img = img.copy()
            for radius in case['radii']:
                cv2.circle(test_img, (center_x, center_y), radius, (255, 255, 255), 2)
            
            self.images.append({
                'image': test_img,
                'mm_per_pixel': 0.1
            })
        
        self.current_image_index = 0
        self.update_display()
        self.update_navigation()
        
        for i, case in enumerate(test_cases):
            self.current_measurement = {
                'center': QPoint(center_x, center_y),
                'type': None,
                'radii': {'inner': None, 'middle': None, 'outer': None}
            }
            
            self.current_measurement['radii']['inner'] = case['radii'][0]
            self.current_measurement['radii']['middle'] = case['radii'][1]
            self.current_measurement['radii']['outer'] = case['radii'][2]
            
            self.current_input.setValue(case['current'])
            
            self.save_measurement()
            
            if i < len(test_cases) - 1:
                self.next_image()
        
        QMessageBox.information(self, 'Success', 'Test data has been loaded. Press Ctrl+S to save measurements.')
