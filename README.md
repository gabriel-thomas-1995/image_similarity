# Image Similarity Search Engine

This project implements an image similarity search engine using FastAPI, FAISS, and PyTorch. The system allows users to upload images and retrieve the most similar images from a pre-indexed dataset.

## Features

- **Image Vectorization**: Converts images into feature vectors using a pre-trained EfficientNet model.
- **Similarity Search**: Uses FAISS to perform fast similarity searches on the image vectors.
- **FastAPI**: Exposes a REST API for querying similar images.

## Modules

### 1. `main.py`

The entry point of the application. This file sets up the FastAPI app and includes the necessary routers. It also handles the application's lifespan events, initializing the image search engine at startup.

### 2. `api.py`

Contains the API endpoints for the application. The main endpoint `/api/similar` allows users to post an image and retrieve a list of the most similar images.

### 3. `models.py`

Defines the data models used in the application. The `Image` model represents an image with a URL.

### 4. `image_vectorizer.py`

Contains the `ImageVectorizer` class, which is responsible for converting images into feature vectors using a pre-trained model. The class handles image preprocessing, vector extraction, and GPU utilization.

### 5. `search_engine.py`

Contains the `SearchEngine` class, which handles the FAISS index. This class is responsible for adding vectors to the index, searching for similar vectors, and saving/loading the index to/from disk.

### 6. `setup_env.sh`

A shell script to set up the Python environment and install all necessary dependencies. This script should be run before starting the application.

## Setup and Installation (Locally)

### Prerequisites

- Python 3.8 or higher
- A GPU with CUDA support (optional, but recommended for faster vectorization)

### Setup Environment

To set up the environment, run the following command:

To have a GPU based environment, you need to run:

```bash
sh setup_env_GPU.sh
```

Otherwise If you want to stick with the CPU version:

```bash
sh setup_env.sh
```

This will create a virtual environment, activate it, and install all required dependencies.

### Running the Application

```bash
uvicorn main:app --reload
```

or

```bash
python main.py
```
