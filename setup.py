from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vit-image-retrieval",
    version="1.2.0",
    author="Dr. Sreenivas Bhattiprolu",
    author_email="pythonformicroscopists@google.com",
    description="A Vision Transformer based image retrieval system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bnsreenu/vit-image-retrieval",
    project_urls={
        "Bug Tracker": "https://github.com/bnsreenu/vit-image-retrieval/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "faiss-cpu>=1.7.4",
        "Pillow>=9.0.0",
        "numpy>=1.20.0",
        "PyQt5>=5.15.0",
    ],
    entry_points={
        'console_scripts': [
            'vit-image-retrieval=vit_image_retrieval.cli:main',
            'vit=vit_image_retrieval.main:main',  # Simplified command line prompt
        ],
        'gui_scripts': [
            'vit-image-retrieval-gui=vit_image_retrieval.main:main',  
        ],
    },
)