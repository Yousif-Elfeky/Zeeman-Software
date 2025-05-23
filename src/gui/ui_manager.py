from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QGroupBox, QDoubleSpinBox, QTableWidget
)
from PyQt6.QtCore import Qt
from src.gui.plot_window import PlotWindow
from src.gui.table_window import TableWindow
from src.gui.results_window import ResultsWindow

class UIManager:
    def __init__(self, main_window_ref):
        self.mw = main_window_ref

    def _create_image_controls_group(self) -> QGroupBox:
        image_group = QGroupBox('Image Controls')
        image_layout = QVBoxLayout()

        load_btn = QPushButton('Load Image')
        load_btn.clicked.connect(self.mw.load_image)
        image_layout.addWidget(load_btn)

        zoom_layout = QHBoxLayout()
        zoom_in_btn = QPushButton('Zoom In')
        zoom_in_btn.clicked.connect(self.mw.zoom_in)
        zoom_out_btn = QPushButton('Zoom Out')
        zoom_out_btn.clicked.connect(self.mw.zoom_out)
        reset_view_btn = QPushButton('Reset View')
        reset_view_btn.clicked.connect(self.mw.reset_view)
        
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
        calibrate_btn.clicked.connect(lambda: self.mw.set_measurement_mode('calibrate'))
        calibration_layout.addWidget(calibrate_btn)
        
        self.mw.calibration_label = QLabel("Scale: Not calibrated")
        calibration_layout.addWidget(self.mw.calibration_label)
        
        return calibration_group

    def _create_measurement_controls_group(self) -> QGroupBox:
        measurement_group = QGroupBox("Measurement Controls")
        measurement_layout = QVBoxLayout(measurement_group)
        
        center_btn = QPushButton("Set Center Point")
        center_btn.clicked.connect(lambda: self.mw.set_measurement_mode('center'))
        measurement_layout.addWidget(center_btn)
        
        radius_layout = QHBoxLayout()
        inner_btn = QPushButton("Measure Inner")
        inner_btn.clicked.connect(lambda: self.mw.set_measurement_mode('inner'))
        middle_btn = QPushButton("Measure Middle")
        middle_btn.clicked.connect(lambda: self.mw.set_measurement_mode('middle'))
        outer_btn = QPushButton("Measure Outer")
        outer_btn.clicked.connect(lambda: self.mw.set_measurement_mode('outer'))
        
        radius_layout.addWidget(inner_btn)
        radius_layout.addWidget(middle_btn)
        radius_layout.addWidget(outer_btn)
        measurement_layout.addLayout(radius_layout)

        auto_detect_label = QLabel("Auto Detect Radii (Define Min and Max Radii):")
        measurement_layout.addWidget(auto_detect_label)
        auto_radius_layout = QHBoxLayout()
        auto_inner_btn = QPushButton("Auto Detect Inner")
        auto_inner_btn.clicked.connect(lambda: self.mw.set_measurement_mode('auto_inner'))
        auto_middle_btn = QPushButton("Auto Detect Middle")
        auto_middle_btn.clicked.connect(lambda: self.mw.set_measurement_mode('auto_middle'))
        auto_outer_btn = QPushButton("Auto Detect Outer")
        auto_outer_btn.clicked.connect(lambda: self.mw.set_measurement_mode('auto_outer'))

        auto_radius_layout.addWidget(auto_inner_btn)
        auto_radius_layout.addWidget(auto_middle_btn)
        auto_radius_layout.addWidget(auto_outer_btn)
        measurement_layout.addLayout(auto_radius_layout)
        
        reset_btn = QPushButton("Reset Measurements")
        reset_btn.clicked.connect(self.mw.reset_measurements)
        measurement_layout.addWidget(reset_btn)
        
        return measurement_group

    def _create_measurements_display_group(self) -> QGroupBox:
        measurements_group = QGroupBox("Measurements")
        measurements_layout = QVBoxLayout(measurements_group)
        
        self.mw.measurements_table = QTableWidget()
        self.mw.measurements_table.setColumnCount(3)
        self.mw.measurements_table.setHorizontalHeaderLabels(['Current (A)', 'B Field (T)', 'Actions'])
        self.mw.measurements_table.horizontalHeader().setStretchLastSection(True)
        measurements_layout.addWidget(self.mw.measurements_table)
        
        return measurements_group

    def _create_results_group(self) -> QGroupBox:
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        params_layout = QHBoxLayout()
        
        current_group = QGroupBox('Magnetic Field')
        current_group_layout = QVBoxLayout()
        
        current_layout = QHBoxLayout()
        current_label = QLabel('Current (A):')
        self.mw.current_input = QDoubleSpinBox()
        self.mw.current_input.setRange(0, 100)
        self.mw.current_input.setDecimals(3)
        self.mw.current_input.setSingleStep(0.1)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.mw.current_input)
        current_group_layout.addLayout(current_layout)
        
        self.mw.calibrate_btn = QPushButton('Calibrate Field')
        self.mw.calibrate_btn.clicked.connect(self.mw.show_calibration)
        current_group_layout.addWidget(self.mw.calibrate_btn)
        
        current_group.setLayout(current_group_layout)
        params_layout.addWidget(current_group)
        
        wavelength_layout = QVBoxLayout()
        wavelength_label = QLabel('Wavelength (nm):')
        self.mw.wavelength_input = QDoubleSpinBox()
        self.mw.wavelength_input.setRange(300, 1000)
        self.mw.wavelength_input.setDecimals(1)
        self.mw.wavelength_input.setValue(643.8)
        wavelength_layout.addWidget(wavelength_label)
        wavelength_layout.addWidget(self.mw.wavelength_input)
        params_layout.addLayout(wavelength_layout)
        
        results_layout.addLayout(params_layout)
        
        self.mw.save_measurement_btn = QPushButton('Save Measurement')
        self.mw.save_measurement_btn.clicked.connect(self.mw.save_measurement)
        results_layout.addWidget(self.mw.save_measurement_btn)
        
        buttons_layout = QHBoxLayout()
        self.mw.calculate_btn = QPushButton('Calculate Results')
        self.mw.calculate_btn.clicked.connect(self.mw.calculate_results)
        buttons_layout.addWidget(self.mw.calculate_btn)
        
        self.mw.show_plot_btn = QPushButton('Show Plot')
        self.mw.show_plot_btn.clicked.connect(self.mw.show_plot)
        buttons_layout.addWidget(self.mw.show_plot_btn)
        
        self.mw.show_table_btn = QPushButton('Show Data Table')
        self.mw.show_table_btn.clicked.connect(self.mw.show_table)
        buttons_layout.addWidget(self.mw.show_table_btn)
        
        self.mw.show_results_btn = QPushButton('Show Results')
        self.mw.show_results_btn.clicked.connect(self.mw.show_results)
        buttons_layout.addWidget(self.mw.show_results_btn)
        results_layout.addLayout(buttons_layout)
        
        self.mw.export_btn = QPushButton('Export to CSV')
        self.mw.export_btn.clicked.connect(self.mw.export_to_csv)
        results_layout.addWidget(self.mw.export_btn)
        
        return results_group

    def setup_layout(self):
        central_widget = QWidget()
        self.mw.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        nav_layout = QHBoxLayout()
        self.mw.prev_image_btn = QPushButton('Previous Image')
        self.mw.prev_image_btn.clicked.connect(self.mw.previous_image)
        self.mw.next_image_btn = QPushButton('Next Image')
        self.mw.next_image_btn.clicked.connect(self.mw.next_image)
        self.mw.image_label = QLabel('No image loaded')
        nav_layout.addWidget(self.mw.prev_image_btn)
        nav_layout.addWidget(self.mw.image_label)
        nav_layout.addWidget(self.mw.next_image_btn)
        main_layout.addLayout(nav_layout)
        
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_container = QWidget()
        image_container_layout = QVBoxLayout(image_container)
        self.mw.image_display = QLabel()
        self.mw.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mw.image_display.mousePressEvent = self.mw.image_clicked
        image_container_layout.addWidget(self.mw.image_display)
        image_container_layout.addStretch()
        image_scroll.setWidget(image_container)
        content_layout.addWidget(image_scroll, 80) 
        
        self.mw.control_scroll = QScrollArea()
        self.mw.control_scroll.setWidgetResizable(True)
        control_width = int(self.mw.width() * 0.2) # Initial width based on 20%
        self.mw.control_scroll.setFixedWidth(control_width)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(5)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        control_layout.addWidget(self._create_image_controls_group())
        control_layout.addWidget(self._create_calibration_group())
        control_layout.addWidget(self._create_measurement_controls_group())
        control_layout.addWidget(self._create_measurements_display_group())
        control_layout.addWidget(self._create_results_group())
        control_layout.addStretch()
        
        self.mw.control_scroll.setWidget(control_panel)
        content_layout.addWidget(self.mw.control_scroll, 20)
        
        self.mw.plot_window = PlotWindow(self)
        self.mw.table_window = TableWindow(self)
        self.mw.results_window = ResultsWindow(self)
