from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QDoubleSpinBox, QTableWidget,
                            QTableWidgetItem, QMessageBox, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class CalibrationWindow(QMainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setWindowTitle('Magnetic Field Calibration')
        self.setGeometry(500, 200, 800, 600)
        self.ui_manager = ui_manager
        
        self.calibration_points = []  
        self.calibration_params = None
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        input_layout = QHBoxLayout()
        
        current_layout = QVBoxLayout()
        current_label = QLabel('Current (A):')
        self.current_input = QDoubleSpinBox()
        self.current_input.setRange(0, 100)
        self.current_input.setDecimals(4)
        self.current_input.setSingleStep(0.1)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_input)
        input_layout.addLayout(current_layout)
        
        field_layout = QVBoxLayout()
        field_label = QLabel('Magnetic Field (Gauss):')
        self.field_input = QDoubleSpinBox()
        self.field_input.setRange(-100000, 100000)
        self.field_input.setDecimals(4)
        self.field_input.setSingleStep(100)
        field_layout.addWidget(field_label)
        field_layout.addWidget(self.field_input)
        input_layout.addLayout(field_layout)
        
        self.add_point_btn = QPushButton('Add Point')
        self.add_point_btn.clicked.connect(self.add_point)
        input_layout.addWidget(self.add_point_btn)
        
        layout.addLayout(input_layout)
        
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Current (A)', 'Field (Gauss)'])
        layout.addWidget(self.table)
        
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        button_layout = QHBoxLayout()
        save_button = QPushButton('Save Plot')
        save_button.clicked.connect(self.save_plot)
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)
        
        self.status_label = QLabel('No calibration data')
        layout.addWidget(self.status_label)
    
    def add_point(self):
        current = self.current_input.value()
        field = self.field_input.value()
        
        self.calibration_points.append((current, field))
        
        self.table.setRowCount(len(self.calibration_points))
        row = len(self.calibration_points) - 1
        self.table.setItem(row, 0, QTableWidgetItem(f"{current:.3f}"))
        self.table.setItem(row, 1, QTableWidgetItem(f"{field:.1f}"))
        
        self.update_plot()
    
    def update_plot(self):
        self.ax.clear()
        
        if len(self.calibration_points) > 0:
            currents, fields = zip(*self.calibration_points)
            
            self.ax.scatter(currents, fields, color='blue', marker='o', s=100, label='Measurements')
            
            if len(self.calibration_points) > 1:
                currents = np.array(currents)
                fields = np.array(fields)
                
                slope, intercept = np.polyfit(currents, fields, 1)
                self.calibration_params = (slope, intercept)
                
                x_line = np.linspace(min(currents), max(currents), 100)
                y_line = slope * x_line + intercept
                
                self.ax.plot(x_line, y_line, 'r-', linewidth=2, 
                           label=f'Fit: {slope:.1f} G/A', alpha=0.7)
                
                self.status_label.setText(
                    f'Calibration: B(Gauss) = {slope:.1f} × I(A) + {intercept:.1f}\n'
                    f'For Tesla: B(T) = {slope*1e-4:.6f} × I(A) + {intercept*1e-4:.6f}'
                )
            
            self.ax.set_xlabel('Current (A)')
            self.ax.set_ylabel('Magnetic Field (Gauss)')
            self.ax.set_title('Magnetic Field Calibration')
            self.ax.grid(True)
            self.ax.legend()
            
        self.canvas.draw()
        
    def save_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            'Save Calibration Plot',
            '',
            'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)',
        )
        if file_name:
            self.figure.savefig(file_name, dpi=300, bbox_inches='tight')
    
    def get_field_for_current(self, current):
        if self.calibration_params is None:
            raise ValueError("No calibration data available")
            
        slope, intercept = self.calibration_params
        field_gauss = slope * current + intercept
        return field_gauss * 1e-4  
