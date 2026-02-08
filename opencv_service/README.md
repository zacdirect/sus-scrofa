# OpenCV Image Analysis Service

Containerized OpenCV service for image manipulation detection and analysis.

## Overview

This service provides REST API endpoints for analyzing images using OpenCV computer vision techniques. It runs as a separate container to isolate OpenCV dependencies from the main SusScrofa application.

## Detection Methods

### 1. Manipulation Detection (Gaussian Blur Difference)
Based on the approach from [this Medium article](https://lotalutfunnahar.medium.com/real-image-identification-using-python-f0dcbd772d05):
- Converts image to grayscale
- Applies Gaussian blur
- Calculates absolute difference between original and blurred
- Identifies anomalous regions through thresholding
- Finds contours indicating manipulated areas

### 2. Noise Pattern Analysis
- Analyzes Laplacian variance (noise estimate)
- Compares noise patterns across image quadrants
- Inconsistent noise suggests manipulation or AI generation
- Natural photos have consistent noise patterns

### 3. JPEG Compression Artifact Detection
- Applies DCT to 8x8 blocks (JPEG's compression blocks)
- Analyzes high-frequency components
- Detects inconsistent compression artifacts
- Manipulated images often show irregular compression

## API Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "opencv-analysis"
}
```

### `POST /analyze`
Full analysis using all detection methods.

**Request:** Multipart form-data with 'image' file OR JSON with base64 image.

**Response:**
```json
{
  "success": true,
  "results": {
    "is_suspicious": false,
    "overall_confidence": 0.15,
    "manipulation_detection": {
      "is_manipulated": false,
      "confidence": 0.23,
      "num_anomalies": 3,
      "anomaly_percentage": 0.45,
      "evidence": "Found 3 anomalous regions covering 0.45% of image"
    },
    "noise_analysis": {
      "overall_noise": 123.45,
      "noise_consistency": 0.89,
      "is_noise_inconsistent": false,
      "quadrant_variances": [120.3, 125.1, 122.8, 124.9],
      "coefficient_variation": 0.12
    },
    "jpeg_artifacts": {
      "has_inconsistent_artifacts": false,
      "confidence": 0.08,
      "compression_variation": 0.45,
      "evidence": "Compression variation coefficient: 0.45"
    }
  }
}
```

### `POST /detect-manipulation`
Manipulation detection only (faster, focused analysis).

## Container Management

### Build the Container
```bash
make opencv-build
```

### Start the Service
```bash
make opencv-start
```

The service will be available at `http://localhost:8080`.

### Stop the Service
```bash
make opencv-stop
```

### View Logs
```bash
make opencv-logs
```

### Test Health
```bash
make opencv-test
# or
curl http://localhost:8080/health
```

### Restart Service
```bash
make opencv-restart
```

### Clean Up
```bash
make opencv-clean  # Removes container and image
```

## Manual Container Operations

### Using Podman
```bash
# Build
cd opencv_service
podman build -t sus_scrofa-opencv:latest .

# Run
podman run -d \
  --name sus_scrofa-opencv \
  -p 8080:8080 \
  sus_scrofa-opencv:latest

# Stop
podman stop sus_scrofa-opencv
podman rm sus_scrofa-opencv
```

### Using Docker
```bash
# Build
cd opencv_service
docker build -t sus_scrofa-opencv:latest .

# Run
docker run -d \
  --name sus_scrofa-opencv \
  -p 8080:8080 \
  sus_scrofa-opencv:latest

# Stop
docker stop sus_scrofa-opencv
docker rm sus_scrofa-opencv
```

## Testing the Service

### Test with curl
```bash
# Health check
curl http://localhost:8080/health

# Analyze an image
curl -X POST http://localhost:8080/analyze \
  -F "image=@/path/to/test_image.jpg"
```

### Test with Python
```python
import requests

# Health check
response = requests.get('http://localhost:8080/health')
print(response.json())

# Analyze image
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/analyze',
        files={'image': ('image.jpg', f, 'image/jpeg')}
    )
    print(response.json())
```

## SusScrofa Plugin Integration

The OpenCV service is automatically integrated with SusScrofa through the `opencv_analysis.py` plugin located in `plugins/analyzer/`.

### Enable in SusScrofa

1. **Start the OpenCV service:**
   ```bash
   make opencv-start
   ```

2. **Start SusScrofa:**
   ```bash
   make run
   ```

3. **Upload an image** - The OpenCV analysis will automatically run if the service is available.

### Plugin Features

- **Automatic service detection** - Plugin checks if service is running
- **Graceful degradation** - If service unavailable, analysis is skipped
- **Comprehensive results** - All three detection methods included in analysis
- **Human-readable summaries** - Results formatted for easy interpretation

### Plugin Results

Results appear in the analysis under `opencv_analysis`:

```python
{
  "opencv_analysis": {
    "enabled": true,
    "is_suspicious": false,
    "overall_confidence": 15.23,
    "verdict": "Authentic",
    "summary": "No significant anomalies detected",
    "manipulation": {
      "detected": false,
      "confidence": 23.4,
      "num_anomalies": 3,
      "anomaly_percentage": 0.45,
      "evidence": "Found 3 anomalous regions..."
    },
    "noise": {
      "inconsistent": false,
      "consistency_score": 89.2,
      "overall_noise_level": 123.45,
      "coefficient_variation": 0.12
    },
    "jpeg_artifacts": {
      "inconsistent": false,
      "confidence": 8.1,
      "compression_variation": 0.45,
      "evidence": "Compression variation coefficient: 0.45"
    }
  }
}
```

## Architecture

```
SusScrofa Main Process (Python 3.13)
    ↓
OpenCV Plugin (plugins/analyzer/opencv_analysis.py)
    ↓ HTTP REST API
OpenCV Service Container (Port 8080)
    ↓
OpenCV Python 3.11 + opencv-python-headless
    ↓
Image Analysis Results
```

### Why Containerized?

- **Dependency isolation** - OpenCV has many system dependencies
- **Version control** - Lock OpenCV to specific tested version
- **Easy deployment** - Single container, no system-wide installs
- **Scalability** - Can run multiple instances if needed
- **Compatibility** - Works with both Podman and Docker

## Configuration

### Environment Variables

- `PORT` - Service port (default: 8080)

### Detection Thresholds

Edit `service.py` to adjust detection sensitivity:

```python
# Manipulation detection
is_manipulated = num_anomalies > 5 and anomaly_percentage > 1.0

# Noise inconsistency
is_inconsistent = cv > 0.5

# JPEG artifacts
has_inconsistent = cv_blocks > 1.0
```

## Performance

- **Health check**: <10ms
- **Full analysis**: 500ms - 2s per image (depends on size)
- **Memory**: ~200MB per container
- **CPU**: Single-threaded, benefits from faster CPUs

## Troubleshooting

### Service won't start
```bash
# Check if port 8080 is in use
sudo lsof -i :8080

# Check container logs
make opencv-logs

# Rebuild container
make opencv-clean
make opencv-build
make opencv-start
```

### Plugin shows "Service not available"
```bash
# Verify service is running
make opencv-test

# Check network connectivity
curl http://localhost:8080/health

# Restart service
make opencv-restart
```

### Analysis returns errors
```bash
# Check service logs
make opencv-logs

# Test with sample image
curl -X POST http://localhost:8080/analyze \
  -F "image=@/path/to/known_good_image.jpg"
```

## References

- **OpenCV Documentation**: https://docs.opencv.org/
- **Medium Article**: https://lotalutfunnahar.medium.com/real-image-identification-using-python-f0dcbd772d05
- **Flask API**: https://flask.palletsprojects.com/

## License

Part of SusScrofa - see `docs/LICENSE.txt` for terms.
