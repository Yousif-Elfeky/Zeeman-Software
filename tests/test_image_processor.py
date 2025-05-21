import sys
import os
import unittest
import numpy as np
import cv2

# Add project root to sys.path to allow 'from src...' imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.image_processor import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        """Set up for each test method."""
        self.processor = ImageProcessor()
        # Create a directory for test images if it doesn't exist
        self.test_images_dir = "test_images_temp" 
        os.makedirs(self.test_images_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary test images and directory
        for item in os.listdir(self.test_images_dir):
            try:
                os.remove(os.path.join(self.test_images_dir, item))
            except OSError as e:
                print(f"Error removing file {item}: {e}")
        try:
            os.rmdir(self.test_images_dir)
        except OSError as e:
            print(f"Error removing directory {self.test_images_dir}: {e}")


    def _create_test_image_with_ring(self, filename, img_size=(200, 200), center=(100, 100), radius=50, ring_thickness=1, noise_level=0):
        """
        Helper function to create a simple image with a ring.
        noise_level: 0 for no noise, 1 for mild salt-and-pepper.
        Returns the image as a numpy array (grayscale).
        """
        img = np.zeros(img_size, dtype=np.uint8) # Grayscale image
        cv2.circle(img, center, radius, 255, ring_thickness) # White ring on black background
        
        if noise_level > 0:
            # Add some salt-and-pepper noise
            row, col = img.shape
            s_vs_p = 0.5 # Ratio of salt to pepper
            amount = 0.005 * noise_level # Proportion of image pixels to replace with noise (very mild)
            
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            img[coords[0], coords[1]] = 255

            # Pepper mode
            num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            img[coords[0], coords[1]] = 0
        
        # The method under test expects a processed image (e.g. grayscale).
        # Saving to disk and reloading is not necessary as detect_spectral_lines takes a numpy array.
        # filepath = os.path.join(self.test_images_dir, filename)
        # cv2.imwrite(filepath, img)
        return img # Return image data directly

    def test_detect_clear_ring_exact_annulus(self):
        center_x, center_y = 50, 50
        radius = 30
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("clear_ring.png", img_size=img_size, center=(center_x,center_y), radius=radius)
        
        # Annulus tightly around the actual ring
        # Note: The detect_spectral_lines method in ImageProcessor has its own HoughCircles parameters.
        # The important part here is that the annulus (minRadius, maxRadius for HoughCircles) is correctly used.
        detected_radius = self.processor.detect_spectral_lines(
            processed_image, 
            center_x, center_y, 
            radius_lower_limit=radius-2, radius_upper_limit=radius+2
        )
        self.assertIsNotNone(detected_radius, "No ring detected in clear_ring_exact_annulus.")
        self.assertAlmostEqual(detected_radius, radius, delta=1, msg="Detected radius not close enough in clear_ring_exact_annulus.")

    def test_detect_ring_at_annulus_edge(self):
        center_x, center_y = 50, 50
        radius = 30
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("edge_ring.png", img_size=img_size, center=(center_x,center_y), radius=radius)

        # Annulus where the ring is right at the lower bound
        detected_radius_lower_edge = self.processor.detect_spectral_lines(
            processed_image, center_x, center_y, 
            radius_lower_limit=radius, radius_upper_limit=radius+5
        )
        self.assertIsNotNone(detected_radius_lower_edge, "No ring detected at lower edge of annulus.")
        self.assertAlmostEqual(detected_radius_lower_edge, radius, delta=1, msg="Detected radius not close enough at lower edge.")

        # Annulus where the ring is right at the upper bound
        detected_radius_upper_edge = self.processor.detect_spectral_lines(
            processed_image, center_x, center_y, 
            radius_lower_limit=radius-5, radius_upper_limit=radius
        )
        self.assertIsNotNone(detected_radius_upper_edge, "No ring detected at upper edge of annulus.")
        self.assertAlmostEqual(detected_radius_upper_edge, radius, delta=1, msg="Detected radius not close enough at upper edge.")

    def test_no_ring_in_annulus(self):
        center_x, center_y = 50, 50
        radius = 30 # Ring is at radius 30
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("no_ring_here.png", img_size=img_size, center=(center_x,center_y), radius=radius)

        # Annulus far from the actual ring
        detected_radius = self.processor.detect_spectral_lines(
            processed_image, center_x, center_y, 
            radius_lower_limit=radius+10, radius_upper_limit=radius+20
        )
        self.assertIsNone(detected_radius, f"Expected no ring, but got radius {detected_radius}.")

    def test_detect_ring_with_noise(self):
        center_x, center_y = 50, 50
        radius = 30
        img_size = (100,100)
        # Create an image with a ring and some noise
        processed_image = self._create_test_image_with_ring("noisy_ring.png", img_size=img_size, center=(center_x,center_y), radius=radius, noise_level=1)
        
        detected_radius = self.processor.detect_spectral_lines(
            processed_image, center_x, center_y, 
            radius_lower_limit=radius-3, radius_upper_limit=radius+3
        )
        self.assertIsNotNone(detected_radius, "Detection failed with noise. HoughCircles params may need tuning or better pre-processing.")
        self.assertAlmostEqual(detected_radius, radius, delta=2, msg="Detected radius not close enough with noise.")

    def test_invalid_annulus_limits_inverted(self):
        center_x, center_y = 50, 50
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("any_ring_inverted.png", img_size=img_size, center=(center_x,center_y), radius=30)

        # Lower limit greater than upper limit
        # The method itself raises ValueError for this.
        with self.assertRaises(ValueError):
            self.processor.detect_spectral_lines(
                processed_image, center_x, center_y, 
                radius_lower_limit=30, radius_upper_limit=20
            )

    def test_invalid_annulus_limits_negative(self):
        center_x, center_y = 50, 50
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("any_ring_negative.png", img_size=img_size, center=(center_x,center_y), radius=30)
        
        # Negative limits
        # The method itself raises ValueError for this.
        with self.assertRaises(ValueError):
            self.processor.detect_spectral_lines(
                processed_image, center_x, center_y, 
                radius_lower_limit=-10, radius_upper_limit=10
            )
        
        with self.assertRaises(ValueError):
            self.processor.detect_spectral_lines(
                processed_image, center_x, center_y, 
                radius_lower_limit=5, radius_upper_limit=-10
            )

    def test_non_grayscale_image_input(self):
        center_x, center_y = 50, 50
        radius = 30
        img_size_color = (100,100,3) # Color image
        # Create a dummy color image (e.g., BGR)
        color_image = np.zeros(img_size_color, dtype=np.uint8)
        cv2.circle(color_image, (center_x, center_y), radius, (255,255,255), 1)

        with self.assertRaisesRegex(ValueError, "Processed image must be grayscale."):
            self.processor.detect_spectral_lines(
                color_image, center_x, center_y,
                radius_lower_limit=radius-2, radius_upper_limit=radius+2
            )

if __name__ == '__main__':
    unittest.main()
