from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class PlotWindow(QMainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setWindowTitle('Energy Shift vs Magnetic Field')
        self.setGeometry(200, 200, 800, 600)
        self.ui_manager = ui_manager
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
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
    
    def plot_data(self, measurements):
        self.ax.clear()
        valid_measurements = [m for m in measurements if m.delta_E_i is not None and m.delta_E_o is not None]
        
        if valid_measurements:
            B_values = [m.B_field for m in valid_measurements]
            E_i_values = [abs(m.delta_E_i) for m in valid_measurements]  
            E_o_values = [abs(m.delta_E_o) for m in valid_measurements]  
            
            self.ax.scatter(B_values, E_i_values, color='blue', marker='o', s=100, label='Inner shifts', zorder=3)
            self.ax.scatter(B_values, E_o_values, color='red', marker='o', s=100, label='Outer shifts', zorder=3)
            
            if len(B_values) > 1:
                B_line = np.linspace(min(B_values), max(B_values), 100)
                
                B_array = np.array(B_values)
                E_i_array = np.array(E_i_values)
                E_o_array = np.array(E_o_values)
                
                z_i = np.polyfit(B_array, E_i_array, 1)
                p_i = np.poly1d(z_i)
                self.ax.plot(B_line, p_i(B_line), 'b-', linewidth=2, label=f'Inner fit: {z_i[0]:.3e} J/T', alpha=0.7, zorder=2)
                
                z_o = np.polyfit(B_array, E_o_array, 1)
                p_o = np.poly1d(z_o)
                self.ax.plot(B_line, p_o(B_line), 'r-', linewidth=2, label=f'Outer fit: {z_o[0]:.3e} J/T', alpha=0.7, zorder=2)
            
            self.ax.set_xlabel('Magnetic Field (T)')
            self.ax.set_ylabel('|Energy Shift| (J)')
            self.ax.set_title('Energy Shift vs Magnetic Field')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()
            
    def save_plot(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            'Save Plot',
            '',
            'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)',
        )
        if file_name:
            self.figure.savefig(file_name, dpi=300, bbox_inches='tight')
            
        self.canvas.draw()
