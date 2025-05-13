from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QDoubleSpinBox, QTableWidget,
                            QTableWidgetItem, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Magnetic Field Calibration')
        self.setGeometry(500, 200, 800, 600)
        
        # Data storage
        self.calibration_points = []  # List of (current, field) tuples
        self.calibration_params = None  # Will store (slope, intercept)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Input section
        input_layout = QHBoxLayout()
        
        # Current input
        current_layout = QVBoxLayout()
        current_label = QLabel('Current (A):')
        self.current_input = QDoubleSpinBox()
        self.current_input.setRange(0, 100)
        self.current_input.setDecimals(4)
        self.current_input.setSingleStep(0.1)
        current_layout.addWidget(current_label)
        current_layout.addWidget(self.current_input)
        input_layout.addLayout(current_layout)
        
        # Field input
        field_layout = QVBoxLayout()
        field_label = QLabel('Magnetic Field (Gauss):')
        self.field_input = QDoubleSpinBox()
        self.field_input.setRange(-100000, 100000)
        self.field_input.setDecimals(4)
        self.field_input.setSingleStep(100)
        field_layout.addWidget(field_label)
        field_layout.addWidget(self.field_input)
        input_layout.addLayout(field_layout)
        
        # Add point button
        self.add_point_btn = QPushButton('Add Point')
        self.add_point_btn.clicked.connect(self.add_point)
        input_layout.addWidget(self.add_point_btn)
        
        layout.addLayout(input_layout)
        
        # Table for points
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Current (A)', 'Field (Gauss)'])
        layout.addWidget(self.table)
        
        # Plot
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Calibration status
        self.status_label = QLabel('No calibration data')
        layout.addWidget(self.status_label)
    
    def add_point(self):
        """Add a calibration point."""
        current = self.current_input.value()
        field = self.field_input.value()
        
        # Add to list
        self.calibration_points.append((current, field))
        
        # Update table
        self.table.setRowCount(len(self.calibration_points))
        row = len(self.calibration_points) - 1
        self.table.setItem(row, 0, QTableWidgetItem(f"{current:.3f}"))
        self.table.setItem(row, 1, QTableWidgetItem(f"{field:.1f}"))
        
        # Update plot and fit
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current data and fit."""
        self.ax.clear()
        
        if len(self.calibration_points) > 0:
            # Split points into x and y arrays
            currents, fields = zip(*self.calibration_points)
            
            # Plot points
            self.ax.scatter(currents, fields, color='blue', marker='o', s=100, label='Measurements')
            
            # Add fit line if we have enough points
            if len(self.calibration_points) > 1:
                # Convert to numpy arrays for fitting
                currents = np.array(currents)
                fields = np.array(fields)
                
                # Fit line
                slope, intercept = np.polyfit(currents, fields, 1)
                self.calibration_params = (slope, intercept)
                
                # Create line points
                x_line = np.linspace(min(currents), max(currents), 100)
                y_line = slope * x_line + intercept
                
                # Plot line
                self.ax.plot(x_line, y_line, 'r-', linewidth=2, 
                           label=f'Fit: {slope:.1f} G/A', alpha=0.7)
                
                # Update status
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
    
    def get_field_for_current(self, current):
        """Convert current to magnetic field in Tesla using calibration."""
        if self.calibration_params is None:
            raise ValueError("No calibration data available")
            
        slope, intercept = self.calibration_params
        field_gauss = slope * current + intercept
        return field_gauss * 1e-4  # Convert Gauss to Tesla
