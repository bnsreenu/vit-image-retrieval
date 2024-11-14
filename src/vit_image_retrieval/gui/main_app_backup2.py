
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import logging
logging.getLogger('faiss.loader').setLevel(logging.WARNING)

import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, QSpinBox, 
                           QProgressBar, QScrollArea, QGridLayout, QMessageBox, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import logging
from pathlib import Path
from datetime import datetime

import torch

from vit_image_retrieval.core.feature_extractor import ImageFeatureExtractor
from vit_image_retrieval.core.retrieval_system import ImageRetrievalSystem
from vit_image_retrieval.core.image_display import EnhancedImageDisplay


# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractionWorker(QThread):
    """Worker thread for feature extraction to prevent GUI freezing"""
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, image_dir, retrieval_system, index_path, metadata_path):
        super().__init__()
        self.image_dir = image_dir
        self.retrieval_system = retrieval_system
        self.index_path = index_path
        self.metadata_path = metadata_path

    def run(self):
        try:
            # Process the images
            self.retrieval_system.index_images(
                self.image_dir,
                progress_callback=self.progress.emit
            )
            
            # Save the results
            self.retrieval_system.save(self.index_path, self.metadata_path)
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class FeatureExtractionTab(QWidget):
    def __init__(self, model_dir=None):
        super().__init__()
        self.model_dir = model_dir
        self.retrieval_system = ImageRetrievalSystem(
            feature_extractor=ImageFeatureExtractor(model_dir=model_dir)
        )
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Directory selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        self.select_dir_btn = QPushButton("Select Image Directory")
        self.select_dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.select_dir_btn)
        layout.addLayout(dir_layout)

        # Index name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Index Name (optional):"))
        self.index_name_input = QLineEdit()
        self.index_name_input.setPlaceholderText("e.g., animals, cars, etc.")
        name_layout.addWidget(self.index_name_input)
        layout.addLayout(name_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Extract button
        self.extract_btn = QPushButton("Extract Features")
        self.extract_btn.clicked.connect(self.start_extraction)
        self.extract_btn.setEnabled(False)
        layout.addWidget(self.extract_btn)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.dir_label.setText(dir_path)
            self.extract_btn.setEnabled(True)
            self.selected_dir = dir_path

    def start_extraction(self):
        if not hasattr(self, 'selected_dir'):
            QMessageBox.warning(self, "Warning", "Please select a directory first.")
            return

        self.extract_btn.setEnabled(False)
        self.select_dir_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Get custom name if provided
        custom_name = self.index_name_input.text().strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if custom_name:
            # Replace spaces and special characters with underscores
            custom_name = ''.join(c if c.isalnum() else '_' for c in custom_name)
            index_path = f"index_{custom_name}_{timestamp}.faiss"
            metadata_path = f"metadata_{custom_name}_{timestamp}.json"
        else:
            index_path = f"index_{timestamp}.faiss"
            metadata_path = f"metadata_{timestamp}.json"
        
        self.worker = FeatureExtractionWorker(
            self.selected_dir, 
            self.retrieval_system,
            index_path,
            metadata_path
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.extraction_finished)
        self.worker.error.connect(self.extraction_error)
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def extraction_finished(self):
        self.status_label.setText(
            f"Features extracted and saved to:\n"
            f"Index: {self.worker.index_path}\n"
            f"Metadata: {self.worker.metadata_path}"
        )
        self.extract_btn.setEnabled(True)
        self.select_dir_btn.setEnabled(True)
        
        QMessageBox.information(
            self, 
            "Success", 
            f"Feature extraction completed successfully!\nIndex saved as: {self.worker.index_path}"
        )

    def extraction_error(self, error_msg):
        self.status_label.setText(f"Error: {error_msg}")
        self.extract_btn.setEnabled(True)
        self.select_dir_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Feature extraction failed: {error_msg}")


class RetrievalTab(QWidget):
    def __init__(self, model_dir=None):
        super().__init__()
        self.model_dir = model_dir
        self.retrieval_system = None
        # Define font sizes as class attributes for easy modification
        self.LARGE_FONT = "font-size: 14px;"
        self.MEDIUM_FONT = "font-size: 12px;"
        self.SMALL_FONT = "font-size: 11px;"
        self.setup_ui()
        self.try_load_latest_index()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Index selection section
        index_section = QWidget()
        index_layout = QVBoxLayout()
        
        # Current index display with larger font
        current_index_layout = QHBoxLayout()
        index_title = QLabel("Current Index:")
        index_title.setStyleSheet(self.LARGE_FONT + "font-weight: bold;")
        
        self.current_index_label = QLabel("No index loaded")
        self.current_index_label.setStyleSheet(f"""
            QLabel {{
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 4px;
                {self.MEDIUM_FONT}
            }}
        """)
        current_index_layout.addWidget(index_title)
        current_index_layout.addWidget(self.current_index_label, stretch=1)
        
        # Index selection buttons with larger font
        buttons_layout = QHBoxLayout()
        self.load_latest_btn = QPushButton("Load Latest Index")
        self.load_specific_btn = QPushButton("Load Specific Index")
        for btn in [self.load_latest_btn, self.load_specific_btn]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 8px;
                    {self.MEDIUM_FONT}
                }}
            """)
        self.load_latest_btn.clicked.connect(self.try_load_latest_index)
        self.load_specific_btn.clicked.connect(self.load_specific_index)
        buttons_layout.addWidget(self.load_latest_btn)
        buttons_layout.addWidget(self.load_specific_btn)
        
        index_layout.addLayout(current_index_layout)
        index_layout.addLayout(buttons_layout)
        index_section.setLayout(index_layout)
        layout.addWidget(index_section)

        # Query image selection with larger font
        query_layout = QHBoxLayout()
        query_title = QLabel("Query Image:")
        query_title.setStyleSheet(self.LARGE_FONT + "font-weight: bold;")
        
        self.query_label = QLabel("No query image selected")
        self.query_label.setStyleSheet(f"""
            QLabel {{
                padding: 8px;
                background-color: #f0f0f0;
                border-radius: 4px;
                {self.MEDIUM_FONT}
            }}
        """)
        self.select_query_btn = QPushButton("Select Query Image")
        self.select_query_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 8px;
                {self.MEDIUM_FONT}
            }}
        """)
        self.select_query_btn.clicked.connect(self.select_query_image)
        query_layout.addWidget(query_title)
        query_layout.addWidget(self.query_label, stretch=1)
        query_layout.addWidget(self.select_query_btn)
        layout.addLayout(query_layout)

        # Number of results selection with larger font
        num_results_layout = QHBoxLayout()
        num_results_label = QLabel("Number of results:")
        num_results_label.setStyleSheet(self.LARGE_FONT + "font-weight: bold;")
        self.num_results_spin = QSpinBox()
        self.num_results_spin.setRange(1, 20)
        self.num_results_spin.setValue(5)
        self.num_results_spin.setStyleSheet(f"""
            QSpinBox {{
                padding: 6px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                {self.MEDIUM_FONT}
            }}
        """)
        num_results_layout.addWidget(num_results_label)
        num_results_layout.addWidget(self.num_results_spin)
        num_results_layout.addStretch()
        layout.addLayout(num_results_layout)

        # Search button with larger font
        self.search_btn = QPushButton("Search")
        self.search_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
                {self.LARGE_FONT}
                font-weight: bold;
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
            }}
        """)
        self.search_btn.clicked.connect(self.perform_search)
        self.search_btn.setEnabled(False)
        layout.addWidget(self.search_btn)

        # Results display
        self.results_scroll = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QGridLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setMinimumHeight(400)
        layout.addWidget(self.results_scroll)

        self.setLayout(layout)


    def try_load_latest_index(self):
        """Try to load the most recent index file from the current directory."""
        try:
            current_dir = Path.cwd()
            faiss_files = list(current_dir.glob("*.faiss"))
            json_files = list(current_dir.glob("*.json"))
            
            if not faiss_files or not json_files:
                self.current_index_label.setText("No index files found")
                return False
                
            # Get the most recent index file
            latest_index = max(faiss_files, key=lambda x: x.stat().st_mtime)
            latest_meta = max(json_files, key=lambda x: x.stat().st_mtime)
            
            return self.load_index_files(str(latest_index), str(latest_meta))
            
        except Exception as e:
            logger.error(f"Error loading latest index: {str(e)}")
            self.current_index_label.setText("Error loading latest index")
            return False

    def load_specific_index(self):
        """Open file dialog to load a specific index file."""
        try:
            index_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Index File", 
                "", 
                "FAISS Files (*.faiss)"
            )
            if not index_path:
                return False
                
            # Try to find corresponding metadata file
            base_name = os.path.splitext(index_path)[0]
            meta_path = f"{base_name}.json"
            
            if not os.path.exists(meta_path):
                # If not found, ask user to select metadata file
                meta_path, _ = QFileDialog.getOpenFileName(
                    self, 
                    "Select Metadata File", 
                    "", 
                    "JSON Files (*.json)"
                )
                if not meta_path:
                    return False
            
            return self.load_index_files(index_path, meta_path)
            
        except Exception as e:
            logger.error(f"Error loading specific index: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load index: {str(e)}")
            return False

    def load_index_files(self, index_path: str, metadata_path: str) -> bool:
        """Load the index and metadata files."""
        try:
            self.retrieval_system = ImageRetrievalSystem(
                feature_extractor=ImageFeatureExtractor(model_dir=self.model_dir),
                index_path=index_path,
                metadata_path=metadata_path
            )
            
            # Update UI
            index_name = os.path.basename(index_path)
            self.current_index_label.setText(
                f"Loaded: {index_name} ({self.retrieval_system.index.ntotal} images)"
            )
            self.search_btn.setEnabled(True)
            return True
            
        except Exception as e:
            logger.error(f"Error loading index files: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load index: {str(e)}")
            return False

    def select_query_image(self):
        """Open file dialog to select query image."""
        image_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Query Image", 
            "", 
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if image_path:
            self.query_label.setText(os.path.basename(image_path))
            self.query_image_path = image_path
            if self.retrieval_system:
                self.search_btn.setEnabled(True)

    def perform_search(self):
        """Execute the search with selected query image."""
        if not hasattr(self, 'query_image_path'):
            QMessageBox.warning(self, "Warning", "Please select a query image first.")
            return

        try:
            results = self.retrieval_system.search(
                self.query_image_path,
                k=self.num_results_spin.value()
            )
            self.display_results(results)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Search failed: {str(e)}")

    def display_results(self, results):
        """Display search results in the UI."""
        # Clear previous results
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)

        # Display query image section
        query_section = QWidget()
        query_layout = QVBoxLayout()
        
        query_label = QLabel("Query Image:")
        query_label.setStyleSheet(self.LARGE_FONT + "font-weight: bold;")
        query_layout.addWidget(query_label)
        
        query_display = EnhancedImageDisplay()
        query_display.set_image(self.query_image_path)
        query_layout.addWidget(query_display)
        query_layout.addStretch()
        
        query_section.setLayout(query_layout)
        self.results_layout.addWidget(query_section, 0, 0)

        # Display results section
        results_section = QWidget()
        results_layout = QVBoxLayout()
        
        results_label = QLabel("Similar Images:")
        results_label.setStyleSheet(self.LARGE_FONT + "font-weight: bold;")
        results_layout.addWidget(results_label)

        # Create grid for results
        results_grid = QGridLayout()
        results_grid.setSpacing(20)
        
        row = 0
        col = 0
        max_cols = 2

        for rank, result in enumerate(results, 1):
            path, similarity, metadata = result
            
            result_widget = QWidget()
            result_layout = QVBoxLayout()
            result_layout.setSpacing(5)
            
            # Add rank number with background
            rank_widget = QWidget()
            rank_layout = QHBoxLayout()
            rank_layout.setContentsMargins(0, 0, 0, 0)
            
            rank_label = QLabel(f"Rank #{rank}")
            rank_label.setStyleSheet(f"""
                QLabel {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 5px 10px;
                    border-radius: 4px;
                    {self.MEDIUM_FONT}
                    font-weight: bold;
                }}
            """)
            rank_layout.addWidget(rank_label)
            rank_layout.addStretch()
            rank_widget.setLayout(rank_layout)
            result_layout.addWidget(rank_widget)
            
            # Image display
            img_display = EnhancedImageDisplay()
            img_display.set_image(path)
            result_layout.addWidget(img_display)
            
            # Info label with larger font
            info_text = (
                f"Similarity: {similarity:.3f}\n"
                f"Distance: {metadata['distance']:.3f}\n"
                f"File: {metadata['filename']}"
            )
            info_label = QLabel(info_text)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet(f"""
                QLabel {{
                    background-color: #f0f0f0;
                    padding: 8px;
                    border-radius: 4px;
                    {self.MEDIUM_FONT}
                }}
            """)
            
            result_layout.addWidget(info_label)
            result_widget.setLayout(result_layout)
            
            results_grid.addWidget(result_widget, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        results_layout.addLayout(results_grid)
        results_layout.addStretch()
        results_section.setLayout(results_layout)
        self.results_layout.addWidget(results_section, 0, 1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ViT Image Retrieval System")
        self.setMinimumSize(800, 600)
        
        # Get default model directory
        self.model_dir = os.getenv('VIT_MODEL_DIR', None)  # Can be set via environment variable
        if self.model_dir:
            logger.info(f"Using custom model directory: {self.model_dir}")
        else:
            self.model_dir = torch.hub.get_dir()
            logger.info(f"Using default model directory: {self.model_dir}")
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.addTab(FeatureExtractionTab(model_dir=self.model_dir), "Feature Extraction")
        tabs.addTab(RetrievalTab(model_dir=self.model_dir), "Image Retrieval")

        self.setCentralWidget(tabs)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()