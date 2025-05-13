from PyQt6.QtWidgets import QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

class TableWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Measurements Data')
        self.setGeometry(300, 300, 1000, 400)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            'B (T)', 'Rc (mm)', 'Ri (mm)', 'Ro (mm)',
            'Δλi (nm)', 'Δλo (nm)', 'ΔEi (eV)', 'ΔEo (eV)'
        ])
        layout.addWidget(self.table)
    
    def update_table(self, measurements):
        """Update the table with measurements data."""
        self.table.setRowCount(len(measurements))
        
        for i, m in enumerate(measurements):
            self.table.setItem(i, 0, QTableWidgetItem(f"{m.B_field:.3f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{m.R_center:.3f}" if m.R_center else ""))
            self.table.setItem(i, 2, QTableWidgetItem(f"{m.R_inner:.3f}" if m.R_inner else ""))
            self.table.setItem(i, 3, QTableWidgetItem(f"{m.R_outer:.3f}" if m.R_outer else ""))
            
            if m.delta_lambda_i is not None:
                self.table.setItem(i, 4, QTableWidgetItem(f"{m.delta_lambda_i*1e9:.3f}"))
            if m.delta_lambda_o is not None:
                self.table.setItem(i, 5, QTableWidgetItem(f"{m.delta_lambda_o*1e9:.3f}"))
            if m.delta_E_i is not None:
                self.table.setItem(i, 6, QTableWidgetItem(f"{m.delta_E_i/1.602176634e-19:.3e}"))
            if m.delta_E_o is not None:
                self.table.setItem(i, 7, QTableWidgetItem(f"{m.delta_E_o/1.602176634e-19:.3e}"))
