from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Energy Shift vs Magnetic Field')
        self.setGeometry(200, 200, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create matplotlib figure and canvas
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
    
    def plot_data(self, measurements):
        """Plot |ΔE| vs B for both inner and outer shifts."""
        self.ax.clear()
        valid_measurements = [m for m in measurements if m.delta_E_i is not None and m.delta_E_o is not None]
        
        if valid_measurements:
            B_values = [m.B_field for m in valid_measurements]
            E_i_values = [abs(m.delta_E_i)/1.602176634e-19 for m in valid_measurements]  # Convert to eV
            E_o_values = [abs(m.delta_E_o)/1.602176634e-19 for m in valid_measurements]  # Convert to eV
            
            # Plot inner shifts as scatter points
            self.ax.scatter(B_values, E_i_values, color='blue', marker='o', s=100, label='Inner shifts', zorder=3)
            # Plot outer shifts as scatter points
            self.ax.scatter(B_values, E_o_values, color='red', marker='o', s=100, label='Outer shifts', zorder=3)
            
            # Add trend lines
            if len(B_values) > 1:
                # Create more points for smooth lines
                B_line = np.linspace(min(B_values), max(B_values), 100)
                
                # Inner trend line
                z_i = np.polyfit(B_values, E_i_values, 1)
                p_i = np.poly1d(z_i)
                self.ax.plot(B_line, p_i(B_line), 'b-', linewidth=2, 
                            label=f'Inner fit: {z_i[0]:.3e} eV/T', alpha=0.7, zorder=2)
                
                # Outer trend line
                z_o = np.polyfit(B_values, E_o_values, 1)
                p_o = np.poly1d(z_o)
                self.ax.plot(B_line, p_o(B_line), 'r-', linewidth=2, 
                            label=f'Outer fit: {z_o[0]:.3e} eV/T', alpha=0.7, zorder=2)
            
            self.ax.set_xlabel('Magnetic Field (T)')
            self.ax.set_ylabel('|Energy Shift| (eV)')
            self.ax.set_title('Energy Shift vs Magnetic Field')
            self.ax.grid(True)
            self.ax.legend()
            
        self.canvas.draw()
