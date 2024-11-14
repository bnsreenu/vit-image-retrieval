"""
Main entry point for the ViT Image Retrieval application.
This module creates and runs the main application window.
"""

import sys
import os
import logging
import platform
from PyQt5.QtWidgets import QApplication
from .gui.main_app import MainWindow

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_platform_specific():
    """Handle platform-specific setup requirements."""
    if platform.system().lower().startswith("linux"):
        logger.info("Linux system detected, adjusting Qt platform settings")
        # Remove potentially conflicting Qt platform plugin path
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        
        # Additional Linux-specific setup can go here
        # For example, setting the style to fusion which often works better on Linux
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
            app.setStyle('Fusion')
    
    elif platform.system().lower() == "darwin":  # macOS
        logger.info("macOS system detected")
        # Add any macOS specific setup here if needed
        os.environ['QT_MAC_WANTS_LAYER'] = '1'  # Helps with some macOS rendering issues
    
    elif platform.system().lower() == "windows":
        logger.info("Windows system detected")
        # Add any Windows specific setup here if needed
        pass

def main():
    """
    Main function to run the ViT Image Retrieval application.
    """
    try:
        # Perform platform-specific setup
        setup_platform_specific()
        
        # Create the application instance if it doesn't exist
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()
        
        # Create and show the main window
        window = MainWindow()
        window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        raise

if __name__ == "__main__":
    main()