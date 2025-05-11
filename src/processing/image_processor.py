import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.image = None
        self.processed_image = None
    
    def load_image(self, file_path):
        """Load and preprocess the image."""
        self.image = cv2.imread(file_path)
        if self.image is None:
            raise ValueError("Failed to load image")
        return self.image
    
    def enhance_image(self):
        """Enhance image quality through noise reduction and contrast adjustment."""
        if self.image is None:
            raise ValueError("No image loaded")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.processed_image = clahe.apply(blurred)
        
        return self.processed_image
    
    def detect_spectral_lines(self):
        """Detect and measure the radii of spectral lines."""
        if self.processed_image is None:
            raise ValueError("No processed image available")
        
        # Implementation of spectral line detection will go here
        # This will include edge detection and line measurement
        pass
