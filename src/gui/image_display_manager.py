import cv2
import numpy as np
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel 
from typing import Optional 

class ImageDisplayManager:
    def __init__(self, image_display_label: QLabel, main_window_ref, ui_manager):
        self.image_display_label = image_display_label 
        self.main_window = main_window_ref 
        self.ui_manager = ui_manager
        
        self.scale_factor = 1.0

    def update_display_pixmap(self, q_img: QImage):
        if self.scale_factor != 1.0:
            if q_img is None: return
            scaled_width = int(q_img.width() * self.scale_factor)
            scaled_height = int(q_img.height() * self.scale_factor)
            if scaled_width > 0 and scaled_height > 0:
                q_img = q_img.scaled(scaled_width, scaled_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                pass 
        
        if q_img:
            self.image_display_label.setPixmap(QPixmap.fromImage(q_img))
        else:
            self.image_display_label.clear()


    def get_current_cv_image_for_display(self) -> Optional[np.ndarray]:
        if not self.main_window.images or self.main_window.current_image_index < 0:
            self.image_display_label.clear() 
            return None
        img_data = self.main_window.images[self.main_window.current_image_index]
        return img_data['image'].copy()

    def convert_cv_to_qimage(self, cv_img: np.ndarray) -> Optional[QImage]:
        if cv_img is None: return None
        
        if cv_img.ndim == 3: 
            height, width, channel = cv_img.shape
            bytes_per_line = channel * width
            if channel == 3: 
                qformat = QImage.Format.Format_RGB888
            else: 
                qformat = QImage.Format.Format_RGBA8888
        elif cv_img.ndim == 2: 
            height, width = cv_img.shape
            bytes_per_line = width
            qformat = QImage.Format.Format_Grayscale8
        else:
            return None 

        return QImage(cv_img.data, width, height, bytes_per_line, qformat)

    def redraw_image_with_overlays(self):
        display_cv_img = self.get_current_cv_image_for_display()
        if display_cv_img is None:
            self.main_window.update_navigation() 
            return

        mw = self.main_window 
        mc = mw.measurement_controller
        
        if mc.calibration_points: 
            for point in mc.calibration_points:
                cv2.circle(display_cv_img, (point.x(), point.y()), 3, (0, 255, 0), -1)

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
            
            if mc.auto_detect_limits and mc.auto_detect_limits.get('lower') is not None and mc.auto_detect_limits.get('upper') is not None:
                cv2.circle(display_cv_img, center_coords, int(mc.auto_detect_limits['lower']), (255, 255, 0), 1) 
                cv2.circle(display_cv_img, center_coords, int(mc.auto_detect_limits['upper']), (0, 255, 255), 1) 
            elif mc.is_defining_annulus and mc.auto_detect_limits and mc.auto_detect_limits.get('lower') is not None:
                cv2.circle(display_cv_img, center_coords, int(mc.auto_detect_limits['lower']), (255, 165, 0), 1) 
        
        q_img = self.convert_cv_to_qimage(display_cv_img)
        self.update_display_pixmap(q_img) 
        
        mw.update_navigation()

    def zoom_in(self):
        self.scale_factor *= 1.2
        self.redraw_image_with_overlays()

    def zoom_out(self):
        self.scale_factor /= 1.2
        self.redraw_image_with_overlays()

    def reset_view(self):
        self.scale_factor = 1.0
        self.redraw_image_with_overlays()

    def get_image_coordinates(self, event_pos: QPoint) -> Optional[QPoint]:
        pixmap = self.image_display_label.pixmap()
        if not pixmap or pixmap.isNull(): return None 
            
        label_size = self.image_display_label.size()
        pixmap_size = pixmap.size() 
        
        x_offset = (label_size.width() - pixmap_size.width()) / 2
        y_offset = (label_size.height() - pixmap_size.height()) / 2
        
        x_rel_pixmap = event_pos.x() - x_offset
        y_rel_pixmap = event_pos.y() - y_offset

        if not self.main_window.images or self.main_window.current_image_index < 0: return None
        original_img_data = self.main_window.images[self.main_window.current_image_index]['image']
        original_height, original_width = original_img_data.shape[:2]

        if pixmap_size.width() == 0 or pixmap_size.height() == 0: return None # Avoid division by zero

        original_x = (x_rel_pixmap / pixmap_size.width()) * original_width
        original_y = (y_rel_pixmap / pixmap_size.height()) * original_height
        
        x_coord = max(0, min(original_width - 1, int(original_x)))
        y_coord = max(0, min(original_height - 1, int(original_y)))
        
        return QPoint(x_coord, y_coord)
