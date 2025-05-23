import csv
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog, QDoubleSpinBox, QApplication, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QShortcut, QKeySequence, QScreen
from typing import Optional, Dict
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
from src.gui.ui_manager import UIManager

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
        
        self.image_processor = ImageProcessor()

        self.mm_per_pixel = None
        self.calibration_distance_mm = 2.0  

        self.measurements = []  
        
        self.ui_manager = UIManager(self)
        self.ui_manager.setup_layout() 
        self.calibration_window = CalibrationWindow(self.ui_manager)
         
        self.image_display_manager = ImageDisplayManager(self.image_display, self, self.ui_manager)
        
        self.measurement_controller = MeasurementController(self, self.ui_manager)

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
            if hasattr(self, 'image_display_manager'): 
                self.image_display_manager.scale_factor = 1.0 
            self.update_display()
            self.update_navigation()
            self.update_measurements_display()
    
    def reset_measurements(self):
        if self.current_image_index >= 0:
            self.measurement_controller.reset_all_measurement_states()
            
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
        
        if self.current_image_index >= 0 and self.measurement_controller.current_measurement:
            self.images[self.current_image_index]['measurement'] = self.measurement_controller.current_measurement.copy()
    
    def set_measurement_mode(self, mode):
        self.measurement_controller.set_mode(mode)
        
        self.update_display()
    
    def save_measurement(self):
        if not self.images or self.current_image_index < 0:
            QMessageBox.warning(self, 'Warning', 'No image loaded')
            return
            
        current_m = self.measurement_controller.current_measurement
        if current_m['center'] is None:
            QMessageBox.warning(self, 'Warning', 'Please set center point first')
            return
            
        if any(v is None for v in current_m['radii'].values()):
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
            R_center=current_m['radii']['middle'] * current_data['mm_per_pixel'] if current_m['radii']['middle'] is not None else None,
            R_inner=current_m['radii']['inner'] * current_data['mm_per_pixel'] if current_m['radii']['inner'] is not None else None,
            R_outer=current_m['radii']['outer'] * current_data['mm_per_pixel'] if current_m['radii']['outer'] is not None else None,
            wavelength=self.wavelength_input.value() * 1e-9  # Convert nm to m
        )
        
        measurement = process_measurement(measurement)
        
        self.measurements.append(measurement)
        
        self.table_window.update_table(self.measurements)
        
        # Reset the measurement state in the controller
        self.measurement_controller.initialize_for_new_measurement() # Or a more specific reset method
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
                self.image_display_manager.scale_factor = 1.0
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

        L = 150  
        n = 1.46 
        
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
                B_field = m.B_field * 1e4 
                
                try:
                    slope, intercept = self.calibration_window.calibration_params
                    current = (B_field - intercept) / slope  
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
            # Directly interact with MeasurementController's state for test setup
            self.measurement_controller.current_measurement = {
                'center': QPoint(center_x, center_y),
                'type': None, # Or a specific type if relevant for the test
                'radii': {
                    'inner': case['radii'][0],
                    'middle': case['radii'][1],
                    'outer': case['radii'][2]
                }
            }
            
            self.current_input.setValue(case['current'])
            
            self.save_measurement() # save_measurement now uses controller's state
            
            if i < len(test_cases) - 1:
                self.next_image()
        
        if self.measurements:
            self.plot_window.plot_data(self.measurements)
        
        QMessageBox.information(self, 'Success', 'Test data has been loaded. Press Ctrl+S to save measurements.')
