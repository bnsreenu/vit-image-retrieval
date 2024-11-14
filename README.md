# ViT Image Retrieval

A Python-based content-based image retrieval (CBIR) system using Vision Transformer (ViT) features and FAISS indexing. This application provides both a graphical user interface (GUI) and programmatic API for indexing and searching similar images.

## Features

- **Vision Transformer Features**: Utilizes ViT-B/16 model pre-trained on ImageNet for robust feature extraction
- **Fast Similarity Search**: Implements FAISS indexing for efficient similarity search
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **User-Friendly GUI**: 
  - Interactive interface for feature extraction and image search
  - Double-click or right-click to open images and containing folders
  - Progress tracking for batch operations
- **Multiple Image Format Support**: Handles PNG, JPG, JPEG, and WebP formats
- **GPU Acceleration**: Optional GPU support for faster processing when available

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/bnsreenu/vit-image-retrieval.git
cd vit-image-retrieval
```

2. Install the package:
```bash
pip install -e .
```

### Install from PyPI

```bash
pip install vit-image-retrieval
```

## Usage

### GUI Application

Launch the application using either command:
```bash
vit
# or
vit-image-retrieval-gui
```

The GUI has two main tabs:

1. **Feature Extraction Tab**:
   - Select a directory of images to index
   - Optionally provide an index name
   - Monitor progress through the progress bar
   - Save index for later use

2. **Image Retrieval Tab**:
   - Load previously created index
   - Select query image
   - Set number of similar images to retrieve
   - View results with similarity scores

### Python API

```python
from vit_image_retrieval import ImageRetrievalSystem

# Initialize the system
retrieval = ImageRetrievalSystem()

# Index a directory of images
retrieval.index_images("path/to/image/directory")

# Save the index for later use
retrieval.save("my_index.faiss", "my_metadata.json")

# Load existing index
retrieval = ImageRetrievalSystem(
    index_path="my_index.faiss",
    metadata_path="my_metadata.json"
)

# Search for similar images
results = retrieval.search("path/to/query/image.jpg", k=5)

# Process results
for path, similarity, metadata in results:
    print(f"Similar image: {path} (similarity: {similarity:.3f})")
```

## Platform-Specific Notes

### Windows
- Supports direct file and folder opening through Windows Explorer
- Uses native file selection dialogs

### macOS
- Uses native macOS commands for file operations
- Finder integration for viewing files and folders
- Requires no additional configuration

### Linux
- Automatically handles Qt platform plugin configurations
- Uses system's default applications for file operations
- Requires X11 or Wayland display server

## Requirements

- torch >= 2.0.0
- torchvision >= 0.15.0
- faiss-cpu >= 1.7.4 (or faiss-gpu for GPU support)
- PyQt5 >= 5.15.0
- Pillow >= 9.0.0
- numpy >= 1.20.0

## Development

To contribute or modify:

1. Fork the repository
2. Create a new branch:
```bash
git checkout -b feature-name
```

3. Make changes and test
4. Submit a pull request

## Common Issues and Solutions

1. **Linux Qt Plugin Error**: Automatically handled by removing conflicting Qt plugin paths
2. **GPU Memory Issues**: Use CPU version if encountering GPU memory problems
3. **File Permission Errors**: Check user permissions for the working directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Vision Transformer implementation from torchvision
- FAISS library from Facebook Research
- PyQt5 for the graphical interface

## Citation

If you use this software in your research, please cite:

```bibtex
@software{vit_image_retrieval,
  title = {ViT Image Retrieval},
  author = {Dr. Sreenivas Bhattiprolu},
  year = {2024},
  url = {https://github.com/bnsreenu/vit-image-retrieval}
}
```

## Support

For support, please:
1. Check the issues page for existing solutions
2. Create a new issue with:
   - Your operating system
   - Python version
   - Complete error message
   - Steps to reproduce the problem