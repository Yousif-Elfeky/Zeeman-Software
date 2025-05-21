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
        detection_result = self.processor.detect_spectral_lines(
            processed_image, 
            initial_center_x=center_x, initial_center_y=center_y, 
            radius_lower_limit=radius-2, radius_upper_limit=radius+2,
            center_search_window_half_size=1 
        )
        self.assertIsNotNone(detection_result, "No ring detected in clear_ring_exact_annulus.")
        detected_x, detected_y, detected_radius = detection_result
        self.assertAlmostEqual(detected_radius, radius, delta=1, msg="Detected radius not close enough in clear_ring_exact_annulus.")
        self.assertAlmostEqual(detected_x, center_x, delta=1, msg="Detected center X not close enough in clear_ring_exact_annulus.")
        self.assertAlmostEqual(detected_y, center_y, delta=1, msg="Detected center Y not close enough in clear_ring_exact_annulus.")

    def test_detect_ring_at_annulus_edge(self):
        center_x, center_y = 50, 50
        radius = 30
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("edge_ring.png", img_size=img_size, center=(center_x,center_y), radius=radius)

        # Annulus where the ring is right at the lower bound
        detection_result_lower = self.processor.detect_spectral_lines(
            processed_image, initial_center_x=center_x, initial_center_y=center_y, 
            radius_lower_limit=radius, radius_upper_limit=radius+5,
            center_search_window_half_size=1
        )
        self.assertIsNotNone(detection_result_lower, "No ring detected at lower edge of annulus.")
        lx, ly, lr = detection_result_lower
        self.assertAlmostEqual(lr, radius, delta=1, msg="Detected radius not close enough at lower edge.")
        self.assertAlmostEqual(lx, center_x, delta=1, msg="Detected center X not close enough at lower edge.")
        self.assertAlmostEqual(ly, center_y, delta=1, msg="Detected center Y not close enough at lower edge.")

        # Annulus where the ring is right at the upper bound
        detection_result_upper = self.processor.detect_spectral_lines(
            processed_image, initial_center_x=center_x, initial_center_y=center_y, 
            radius_lower_limit=radius-5, radius_upper_limit=radius,
            center_search_window_half_size=1
        )
        self.assertIsNotNone(detection_result_upper, "No ring detected at upper edge of annulus.")
        ux, uy, ur = detection_result_upper
        self.assertAlmostEqual(ur, radius, delta=1, msg="Detected radius not close enough at upper edge.")
        self.assertAlmostEqual(ux, center_x, delta=1, msg="Detected center X not close enough at upper edge.")
        self.assertAlmostEqual(uy, center_y, delta=1, msg="Detected center Y not close enough at upper edge.")

    def test_no_ring_in_annulus(self):
        center_x, center_y = 50, 50
        radius = 30 # Ring is at radius 30
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("no_ring_here.png", img_size=img_size, center=(center_x,center_y), radius=radius)

        # Annulus far from the actual ring
        detection_result = self.processor.detect_spectral_lines(
            processed_image, initial_center_x=center_x, initial_center_y=center_y, 
            radius_lower_limit=radius+10, radius_upper_limit=radius+20,
            center_search_window_half_size=1
        )
        self.assertIsNone(detection_result, f"Expected no ring, but got result {detection_result}.")

    def test_detect_ring_with_noise(self):
        center_x, center_y = 50, 50
        radius = 30
        img_size = (100,100)
        # Create an image with a ring and some noise
        processed_image = self._create_test_image_with_ring("noisy_ring.png", img_size=img_size, center=(center_x,center_y), radius=radius, noise_level=1)
        
        detection_result = self.processor.detect_spectral_lines(
            processed_image, initial_center_x=center_x, initial_center_y=center_y, 
            radius_lower_limit=radius-3, radius_upper_limit=radius+3,
            center_search_window_half_size=2 
        )
        self.assertIsNotNone(detection_result, "Detection failed with noise. HoughCircles params may need tuning or better pre-processing.")
        detected_x, detected_y, detected_radius = detection_result
        self.assertAlmostEqual(detected_radius, radius, delta=2, msg="Detected radius not close enough with noise.")
        self.assertAlmostEqual(detected_x, center_x, delta=3, msg="Detected center X not close enough with noise.") # Allow larger delta for center with noise
        self.assertAlmostEqual(detected_y, center_y, delta=3, msg="Detected center Y not close enough with noise.")


    def test_invalid_annulus_limits_inverted(self):
        center_x, center_y = 50, 50
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("any_ring_inverted.png", img_size=img_size, center=(center_x,center_y), radius=30)

        # Lower limit greater than upper limit
        # The method itself raises ValueError for this.
        with self.assertRaises(ValueError):
            self.processor.detect_spectral_lines(
                processed_image, initial_center_x=center_x, initial_center_y=center_y, 
                radius_lower_limit=30, radius_upper_limit=20,
                center_search_window_half_size=0
            )

    def test_invalid_annulus_limits_negative(self):
        center_x, center_y = 50, 50
        img_size = (100,100)
        processed_image = self._create_test_image_with_ring("any_ring_negative.png", img_size=img_size, center=(center_x,center_y), radius=30)
        
        # Negative limits
        # The method itself raises ValueError for this.
        with self.assertRaises(ValueError):
            self.processor.detect_spectral_lines(
                processed_image, initial_center_x=center_x, initial_center_y=center_y, 
                radius_lower_limit=-10, radius_upper_limit=10,
                center_search_window_half_size=0
            )
        
        with self.assertRaises(ValueError):
            self.processor.detect_spectral_lines(
                processed_image, initial_center_x=center_x, initial_center_y=center_y, 
                radius_lower_limit=5, radius_upper_limit=-10,
                center_search_window_half_size=0
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
                color_image, initial_center_x=center_x, initial_center_y=center_y,
                radius_lower_limit=radius-2, radius_upper_limit=radius+2,
                center_search_window_half_size=1
            )

    def test_detect_ring_with_offset_center(self):
        initial_center_x, initial_center_y = 50, 50
        actual_ring_center_x, actual_ring_center_y = 52, 53 # Ring is slightly offset
        radius = 25
        img_size = (100,100)
        
        # Create image with the ring at its actual offset center
        # Filename is not strictly needed as we pass image data directly
        processed_image = self._create_test_image_with_ring(
            "offset_center_ring.png", 
            img_size=img_size, 
            center=(actual_ring_center_x, actual_ring_center_y), 
            radius=radius
        )
        
        # Annulus is defined around the *initial* (slightly incorrect) center
        # Center search window should be large enough to find the actual center
        search_window = 5 
        detection_result = self.processor.detect_spectral_lines(
            processed_image, 
            initial_center_x=initial_center_x, initial_center_y=initial_center_y, 
            radius_lower_limit=radius-3, radius_upper_limit=radius+3,
            center_search_window_half_size=search_window 
        )
        
        self.assertIsNotNone(detection_result, "Ring not detected with offset center.")
        detected_x, detected_y, detected_radius = detection_result
        
        self.assertAlmostEqual(detected_radius, radius, delta=1)
        # Check if the detected center is the *actual* ring center, not the initial one
        self.assertAlmostEqual(detected_x, actual_ring_center_x, delta=1, msg="Detected center X is not the actual ring center.")
        self.assertAlmostEqual(detected_y, actual_ring_center_y, delta=1, msg="Detected center Y is not the actual ring center.")
        
        # Check that detected center is different from initial if offset is significant enough
        # (allowing for delta=1 in previous assertions)
        offset_x_significant = abs(actual_ring_center_x - initial_center_x) > 1
        offset_y_significant = abs(actual_ring_center_y - initial_center_y) > 1
        if offset_x_significant or offset_y_significant:
            self.assertTrue(detected_x != initial_center_x or detected_y != initial_center_y, 
                            f"Center should have been adjusted from ({initial_center_x},{initial_center_y}) to ({detected_x},{detected_y}) but wasn't, or was adjusted back to initial.")


if __name__ == '__main__':
    unittest.main()
