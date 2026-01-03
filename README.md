# Revelation

FFXIV equipment recognition microservice by AetherSight.

## Features

Deep learning-based web service for recognizing Final Fantasy XIV equipment from images. Returns top-K recognition results with similarity scores.

- FastAPI-based REST API with automatic OpenAPI documentation
- Efficient embedding-based similarity search
- Automatic gallery cache management
- Feedback system with flexible storage backend (local filesystem or Tencent COS)
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
   - `aethersight.pth` - Main model (required)
   - `aethersight_gallery.pth` - Gallery cache (auto-generated if not exists)

3. Set environment variables (optional):
   - `MODEL_DIR` - Model directory (default: `models`)
   - `GALLERY_ROOT` - Gallery image root directory (only needed if cache doesn't exist)
   - `PORT` - Service port (default: `5000`)
   - `DEBUG` - Debug mode (default: `true`)
   
   **Feedback Storage Configuration:**
   - `STORAGE_TYPE` - Storage type: `local` or `cos` (default: `local`)
   - `FEEDBACK_STORAGE_DIR` - Local storage directory (default: `feedback_images`)
   - `FEEDBACK_DB_PATH` - Database path for feedback records (default: `feedback.db`)
   
   **Tencent COS Configuration (when STORAGE_TYPE=cos):**
   - `COS_SECRET_ID` - Tencent Cloud SecretId (required)
   - `COS_SECRET_KEY` - Tencent Cloud SecretKey (required)
   - `COS_REGION` - COS region (required, e.g., `ap-beijing`)
   - `COS_BUCKET` - COS bucket name (required)
   - `COS_BASE_PATH` - Base path in COS (default: `feedback`)

4. Run:
```bash
poetry run python -m revelation
```

The service will automatically:
- Load the model from `models/aethersight.pth`
- Load gallery cache from `models/aethersight_gallery.pth` (if exists)
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
  - Returns: Top-10 recognition results (display count controlled by frontend)
- `POST /feedback` - Submit feedback with correct label
  - Parameters:
    - `image`: User-marked image region (multipart/form-data)
    - `label`: Correct equipment label (form field)
  - Response:
    - `status`: "success"

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

**Note**: If `aethersight_gallery.pth` exists in the mounted `models/` directory, `GALLERY_ROOT` is not required. Otherwise, mount the gallery directory and set the environment variable:

```bash
docker run -d -p 5000:5000 \
  -v /path/to/models:/app/models \
  -v /path/to/gallery:/path/to/gallery \
  -e GALLERY_ROOT=/path/to/gallery \
  --name revelation \
  revelation
```

**With Feedback Storage (Local):**
```bash
docker run -d -p 5000:5000 \
  -v /path/to/models:/app/models \
  -v /path/to/feedback:/app/feedback_images \
  -v /path/to/feedback.db:/app/feedback.db \
  -e STORAGE_TYPE=local \
  -e FEEDBACK_STORAGE_DIR=/app/feedback_images \
  --name revelation \
  revelation
```

**With Feedback Storage (Tencent COS):**
```bash
docker run -d -p 5000:5000 \
  -v /path/to/models:/app/models \
  -e STORAGE_TYPE=cos \
  -e COS_SECRET_ID=your_secret_id \
  -e COS_SECRET_KEY=your_secret_key \
  -e COS_REGION=ap-beijing \
  -e COS_BUCKET=your-bucket-name \
  --name revelation \
  revelation
```
