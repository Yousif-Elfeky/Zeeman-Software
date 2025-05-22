from PyQt6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea,
                             QPushButton, QHBoxLayout, QApplication)
from PyQt6.QtCore import Qt

class ResultsWindow(QMainWindow):
    def __init__(self, ui_manager):
        super().__init__()
        self.setWindowTitle('Zeeman Effect Results')
        self.setGeometry(400, 400, 600, 400)
        self.ui_manager = ui_manager
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        self.results_label = QLabel()
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self.results_label)
        
        button_layout = QHBoxLayout()
        copy_button = QPushButton('Copy Results')
        copy_button.clicked.connect(self.copy_results)
        button_layout.addStretch()
        button_layout.addWidget(copy_button)
        layout.addLayout(button_layout)
    
    def update_results(self, results):
        bohr_magneton_inner, bohr_magneton_outer, bohr_magneton_avg, \
        specific_charge_inner, specific_charge_outer, specific_charge_avg = results
        
        text = f"Final Results:\n\n"
        text += f"Inner Bohr magneton: {abs(bohr_magneton_inner):.3e} J/T\n"
        text += f"Outer Bohr magneton: {abs(bohr_magneton_outer):.3e} J/T\n"
        text += f"Average Bohr magneton: {bohr_magneton_avg:.3e} J/T\n\n"
        text += f"Inner specific charge (e/m): {specific_charge_inner:.3e} C/kg\n"
        text += f"Outer specific charge (e/m): {specific_charge_outer:.3e} C/kg\n"
        text += f"Average specific charge (e/m): {specific_charge_avg:.3e} C/kg\n\n"
        text += f"Expected values:\n"
        text += f"Bohr magneton: 9.274e-24 J/T\n"
        text += f"Specific charge: 1.758e11 C/kg\n\n"
        text += f"Relative errors:\n"
        text += f"Bohr magneton: {abs(bohr_magneton_avg - 9.274e-24)/9.274e-24*100:.1f}%\n"
        text += f"Specific charge: {abs(specific_charge_avg - 1.758e11)/1.758e11*100:.1f}%"
        
        self.results_label.setText(text)
        self._current_text = text  
        
    def copy_results(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self._current_text)
