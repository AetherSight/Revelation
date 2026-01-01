# Revelation

FFXIV equipment recognition microservice by AetherSight.

## Features

Deep learning-based web service for recognizing Final Fantasy XIV equipment from images. Returns top-K recognition results with similarity scores.

- FastAPI-based REST API with automatic OpenAPI documentation
- Efficient embedding-based similarity search
- Automatic gallery cache management
- Docker-ready for microservice deployment

## Development

### Requirements

- Python 3.10+
- Poetry

### Setup

1. Install dependencies:
```bash
poetry install
```

2. Place model files in `models/`:
   - `epoch_50_supcon.pth` - Main model (required)
   - `epoch_50_supcon_gallery.pth` - Gallery cache (auto-generated if not exists)

3. Set environment variables (optional):
   - `MODEL_DIR` - Model directory (default: `models`)
   - `GALLERY_ROOT` - Gallery image root directory (only needed if cache doesn't exist)
   - `PORT` - Service port (default: `5000`)

4. Run:
```bash
poetry run python -m revelation
```

The service will automatically:
- Load the model from `models/epoch_50_supcon.pth`
- Load gallery cache from `models/epoch_50_supcon_gallery.pth` (if exists)
- Build gallery from `GALLERY_ROOT` if cache doesn't exist
- Start the web server after everything is loaded

## API

Once the service is running, you can access:

- **API Documentation**: http://localhost:5000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:5000/redoc (ReDoc)

### Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict equipment from uploaded image
  - Parameters:
    - `image`: Image file (multipart/form-data)
    - `top_k`: Number of top results to return (default: 5)

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
  --name revelation \
  revelation
```

**Note**: If `epoch_50_supcon_gallery.pth` exists in the mounted `models/` directory, `GALLERY_ROOT` is not required. Otherwise, mount the gallery directory and set the environment variable:

```bash
docker run -d -p 5000:5000 \
  -v /path/to/models:/app/models \
  -v /path/to/gallery:/path/to/gallery \
  -e GALLERY_ROOT=/path/to/gallery \
  --name revelation \
  revelation
```
