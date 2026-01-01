# Revelation

FFXIV equipment recognition microservice by AetherSight.

## Features

Deep learning-based web service for recognizing Final Fantasy XIV equipment from images. Returns top-K recognition results with similarity scores.

## Development

### Requirements

- Python 3.10+
- Poetry

### Setup

1. Initialize Poetry (if not already done):
```bash
poetry init
```

2. Install dependencies:
```bash
poetry install
```

3. Place model files in `models/`:
   - `epoch_50_supcon.pth` - Main model
   - `epoch_50_supcon_gallery.pth` - Gallery cache (auto-generated)

4. Set environment variables:
   - `MODEL_DIR` - Model directory (default: `models`)
   - `GALLERY_ROOT` - Gallery image root directory (optional)
   - `PORT` - Service port (default: `5000`)

5. Run:
```bash
poetry run python -m revelation
```

## Deployment

### Docker

Build:
```bash
docker build -t revelation .
```

Run:
```bash
docker run -d -p 5000:5000 \
  -v /path/to/models:/app/models \
  -v /path/to/gallery:/path/to/gallery \
  -e GALLERY_ROOT=/path/to/gallery \
  --name revelation \
  revelation
```
