#!/usr/bin/env python3
import sys
import os

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(src_dir)
sys.path.insert(0, project_dir)

from PyQt6.QtWidgets import QApplication, QMainWindow
from src.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
