from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMessageBox, QInputDialog
from typing import Optional, Dict, List, Any
import numpy as np

class MeasurementController:
    def __init__(self, main_window_instance, ui_manager): 
        self.mw = main_window_instance 
        self.ui_manager = ui_manager

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
        current_center = self.current_measurement.get('center') 
        self.current_measurement = {
            'center': current_center, 'type': None, 'radii': {'inner': None, 'middle': None, 'outer': None}
        }
        self.current_mode = None
        self.auto_detect_limits = {'lower': None, 'upper': None}
        self.is_defining_annulus = False

    def set_mode(self, mode: Optional[str]):
        intended_mode = mode 

        if intended_mode == 'center':
            self.initialize_for_new_measurement()
            self.current_mode = 'center'         
        elif intended_mode and intended_mode.startswith('auto_'):
            if self.current_measurement.get('center') is None:
                QMessageBox.information(self.mw, 'Set Center First',
                                        'Please set the center point before defining an annulus for auto-detection.')
                self.current_mode = None
                self.is_defining_annulus = False
                self.auto_detect_limits = {'lower': None, 'upper': None}
                self.mw.update_display() 
                return
            else:
                self.current_mode = intended_mode
                self.is_defining_annulus = True
                self.auto_detect_limits = {'lower': None, 'upper': None} 
        else:
            self.current_mode = intended_mode
            self.is_defining_annulus = False 
            self.auto_detect_limits = {'lower': None, 'upper': None} 

        self.mw.update_display() 


    def handle_image_click(self, pos: QPoint):
        if self.mw.current_image_index < 0 or not self.mw.images:
            QMessageBox.warning(self.mw, "No Image", "Please load an image first.")
            return

        current_image_data = self.mw.images[self.mw.current_image_index]

        if self.is_defining_annulus and self.current_mode and self.current_mode.startswith('auto_'):
            center_point = self.current_measurement.get('center')
            if center_point is None: 
                QMessageBox.warning(self.mw, "Error", "Center point is not set. Cannot define annulus.")
                self._reset_auto_detect_state_and_update_ui() 
                return

            dx = pos.x() - center_point.x()
            dy = pos.y() - center_point.y()
            clicked_radius = np.sqrt(dx**2 + dy**2) 

            if self.auto_detect_limits.get('lower') is None:
                self.auto_detect_limits['lower'] = clicked_radius
            else:
                self.auto_detect_limits['upper'] = clicked_radius
                if self.auto_detect_limits['lower'] > self.auto_detect_limits['upper']:
                    self.auto_detect_limits['lower'], self.auto_detect_limits['upper'] = \
                        self.auto_detect_limits['upper'], self.auto_detect_limits['lower']
                
                self.mw.update_display() 

                ring_type_to_update = self.current_mode.split('_')[1]
                
                try:
                    raw_rgb_image = current_image_data['image']
                    self.mw.image_processor.image = raw_rgb_image 
                    enhanced_image = self.mw.image_processor.enhance_image() 
                    
                    initial_center_x = center_point.x()
                    initial_center_y = center_point.y()
                    lower_rad = int(self.auto_detect_limits['lower'])
                    upper_rad = int(self.auto_detect_limits['upper'])

                    center_search_window_size = 10  
                    
                    detected_info_dict = self.mw.image_processor.auto_detect_radius_refined(
                        enhanced_image, initial_center_x, initial_center_y, 
                        lower_rad, upper_rad, center_search_window_half_size=center_search_window_size)

                    if detected_info_dict:
                        det_x = detected_info_dict['center_x']
                        det_y = detected_info_dict['center_y']
                        det_r_centerline = detected_info_dict['radius_centerline']
                        det_r = det_r_centerline 
                        
                        original_center_qpoint = center_point
                        new_center_qpoint = QPoint(det_x, det_y)
                        
                        center_shift_distance = np.sqrt((det_x - original_center_qpoint.x())**2 + 
                                                     (det_y - original_center_qpoint.y())**2)
                        
                        self.current_measurement['center'] = new_center_qpoint
                        self.current_measurement['radii'][ring_type_to_update] = det_r
                        self.current_measurement['type'] = ring_type_to_update
                        
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
                    else:
                        self.current_measurement['radii'][ring_type_to_update] = None
                        QMessageBox.warning(self.mw, 'Failure', f"Auto-detection failed for {ring_type_to_update} ring.")
                except Exception as e:
                    if ring_type_to_update: self.current_measurement['radii'][ring_type_to_update] = None
                    QMessageBox.critical(self.mw, "Processing Error", f"Error during {ring_type_to_update} ring detection: {str(e)}")
                finally:
                    self._reset_auto_detect_state_and_update_ui()
            
            self.mw.update_display() 

        elif self.current_mode == 'calibrate':
            self.calibration_points.append(pos)
            if len(self.calibration_points) == 2:
                calib_dist_mm_val = self.mw.calibration_distance_mm 
                distance, ok = QInputDialog.getDouble(self.mw, 'Enter Distance', 
                                                         'Distance between points (mm):', value=calib_dist_mm_val, 
                                                         min=0.1, max=1000.0, decimals=2)
                if ok:
                    dx = self.calibration_points[1].x() - self.calibration_points[0].x()
                    dy = self.calibration_points[1].y() - self.calibration_points[0].y()
                    pixel_dist = np.sqrt(dx**2 + dy**2) 
                    if pixel_dist > 0:
                        current_image_data['mm_per_pixel'] = distance / pixel_dist
                    else:
                        QMessageBox.warning(self.mw, "Calibration Error", "Calibration points are identical.")
                    self.mw.update_scale_display() 
                self.calibration_points = [] 
                self.current_mode = None 
            self.mw.update_display()

        elif self.current_mode == 'center':
            self.current_measurement['center'] = pos
            self.current_mode = None 
            self.mw.update_display()
            if hasattr(self.mw, 'update_measurements_display'): 
                 self.mw.update_measurements_display()

        elif self.current_mode in ['inner', 'middle', 'outer']:
            center_point = self.current_measurement.get('center')
            if center_point is None:
                QMessageBox.warning(self.mw, 'Warning', 'Please set center point first.')
                self.current_mode = None
                return
            
            dx = pos.x() - center_point.x()
            dy = pos.y() - center_point.y()
            radius = np.sqrt(dx**2 + dy**2) 
            
            self.current_measurement['radii'][self.current_mode] = radius
            self.current_measurement['type'] = self.current_mode
            
            if 'measurement' not in current_image_data or current_image_data['measurement'] is None:
                 current_image_data['measurement'] = {'center': None, 'type': None, 'radii': {'inner':None,'middle':None,'outer':None}}
            current_image_data['measurement']['center'] = self.current_measurement['center']
            current_image_data['measurement']['radii'][self.current_mode] = radius
            current_image_data['measurement']['type'] = self.current_mode

            self.current_mode = None
            self.mw.update_display()
            if hasattr(self.mw, 'update_measurements_display'):
                 self.mw.update_measurements_display()
        else:
            pass

    def _reset_auto_detect_state_and_update_ui(self):
        self.current_mode = None 
        self.auto_detect_limits = {'lower': None, 'upper': None}
        self.is_defining_annulus = False
        self.mw.update_display()
        if hasattr(self.mw, 'update_measurements_display'):
            self.mw.update_measurements_display()
