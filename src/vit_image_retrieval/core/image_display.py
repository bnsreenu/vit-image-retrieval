# image_display.py

import os
import platform
import subprocess
from PyQt5.QtWidgets import QLabel, QMenu, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QMouseEvent, QContextMenuEvent
import logging

logger = logging.getLogger(__name__)

class EnhancedImageDisplay(QLabel):
    """Enhanced image display widget with double-click and context menu support."""
    
    doubleClicked = pyqtSignal(str)
    
    def __init__(self, min_size: int = 200):
        super().__init__()
        self.image_path = None
        self.setMinimumSize(min_size, min_size)
        self.setMaximumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background-color: #f5f5f5;
                padding: 5px;
            }
            QLabel:hover {
                border: 1px solid #999999;
            }
        """)
        self.original_pixmap = None
        self.setContentsMargins(5, 5, 5, 5)

    def open_file(self, file_path: str) -> bool:
        """
        Open a file using the system's default application.
        Returns True if successful, False otherwise.
        """
        try:
            system = platform.system().lower()
            
            if system == 'darwin':  # macOS
                subprocess.run(['open', file_path], check=True)
            elif system == 'windows':
                os.startfile(file_path)
            else:  # Linux and others
                subprocess.run(['xdg-open', file_path], check=True)
            return True
            
        except Exception as e:
            logger.error(f"Error opening file {file_path}: {e}")
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open file: {file_path}\nError: {str(e)}"
            )
            return False

    def open_folder(self, file_path: str) -> bool:
        """
        Open the folder containing the file.
        Returns True if successful, False otherwise.
        """
        try:
            folder_path = os.path.dirname(os.path.abspath(file_path))
            system = platform.system().lower()
            
            if system == 'darwin':  # macOS
                # Use AppleScript to open folder and select file
                script = f'tell application "Finder" to reveal POSIX file "{file_path}"'
                subprocess.run(['osascript', '-e', script], check=True)
                
                script = 'tell application "Finder" to activate'
                subprocess.run(['osascript', '-e', script], check=True)
            elif system == 'windows':
                # Correct syntax for Windows explorer
                file_path = os.path.normpath(file_path)  # Normalize path
                try:
                    # First try to highlight the file
                    subprocess.run(['explorer', '/select,', file_path], shell=True)
                except Exception as e:
                    logger.warning(f"Could not select file, opening folder instead: {e}")
                    # If that fails, just open the folder
                    os.startfile(folder_path)
            else:  # Linux and others
                subprocess.run(['xdg-open', folder_path], check=True)
            return True
            
        except Exception as e:
            logger.error(f"Error opening folder for {file_path}: {e}")
            # On Windows, if both methods fail, try one last time to just open the folder
            if platform.system().lower() == 'windows':
                try:
                    os.startfile(folder_path)
                    return True
                except Exception as nested_e:
                    logger.error(f"Final attempt to open folder failed: {nested_e}")
            
            QMessageBox.warning(
                self,
                "Error",
                f"Could not open folder for: {file_path}\nError: {str(e)}"
            )
            return False

    def set_image(self, image_path: str):
        """Set the image to display."""
        try:
            self.image_path = os.path.abspath(image_path)
            pixmap = QPixmap(image_path)
            self.original_pixmap = pixmap
            scaled_pixmap = pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            self.setText("Error loading image")
            self.image_path = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double-click events."""
        if self.image_path:
            self.open_file(self.image_path)
            self.doubleClicked.emit(self.image_path)

    def contextMenuEvent(self, event: QContextMenuEvent):
        """Handle right-click events."""
        if self.image_path:
            menu = QMenu(self)
            open_action = menu.addAction("Open Image")
            open_folder_action = menu.addAction("Open Containing Folder")
            
            action = menu.exec_(event.globalPos())
            
            if action == open_action:
                self.open_file(self.image_path)
            elif action == open_folder_action:
                self.open_folder(self.image_path)

    def resizeEvent(self, event):
        """Handle resize events to maintain aspect ratio."""
        super().resizeEvent(event)
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)