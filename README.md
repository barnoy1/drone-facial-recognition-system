# Drone Facial Recognition System

A facial recognition system that can process both video streams and image folders, with support for the Labeled Faces in the Wild (LFW) dataset.

## Features

- Real-time facial recognition from camera stream
- Batch processing of images from a folder
- Support for Labeled Faces in the Wild dataset
- Configurable reference images and embeddings storage

## Usage

### Camera Stream Mode

```bash
python src/main.py --mode stream
```

### Folder Processing Mode

```bash
python src/main.py --mode folder --input /path/to/images
```

### Using LFW Dataset

1. Download target person and test images:
```bash
python src/utils/lfw_downloader.py
```

2. Run recognition on the downloaded images:
```bash
python src/main.py --mode folder --input lfw_dataset --reference-dir lfw_dataset/target
```

## Requirements

See requirements.txt for a complete list of dependencies.

## License

MIT License - See LICENSE file for details.