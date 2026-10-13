# WildTrace RoVF Inference Backend

FastAPI inference service for video-based wildlife re-identification based on the research paper:
> **Recurrence over Video Frames (RoVF) for Animal Re-identification**  
> *International Journal of Computer Vision (IJCV), 2026*  
> DOI: [10.1007/s11263-025-02709-8](https://doi.org/10.1007/s11263-025-02709-8)  
> Official Codebase: [Strong-AI-Lab/RoVF-Meerkat-Reidentification](https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification)

---

## Architecture Overview

The inference pipeline strictly adheres to the RoVF architecture:
1. **Spatial Feature Extraction**: Foundation vision transformer backbone (`facebook/dinov2-small`, 384 hidden dim, 257 sequence tokens per frame).
2. **Recurrent State Tracking**: Perceiver recurrent head with cross-attention from learned latents ($257 \times 384$) to frame feature tokens, followed by a 2-layer GELU Transformer Encoder.
3. **Residual Aggregation**: Aggregates recurrent state with average video frame tokens (`is_append_avg_emb=True`, matching the paper's best performing `RoVF-S-af` configuration).
4. **Gallery Retrieval**: Computes true cosine distance against reference gallery embeddings ($E_{query} \cdot E_{gallery}^T$).

---

## Requirements & Installation

Python 3.10+ is required.

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Inference Server

```bash
# Run locally on port 8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

---

## Checkpoint Configuration

By default, the service loads the official `facebook/dinov2-small` backbone from Hugging Face and initializes the RoVF Perceiver recurrent head.

If you have trained a fine-tuned RoVF checkpoint using the official training pipeline:
1. Place your `.pt` checkpoint file in `backend/checkpoints/checkpoint.pt`
2. Or specify its path via the `ROVF_CHECKPOINT_PATH` environment variable:
   ```bash
   # Windows PowerShell:
   $env:ROVF_CHECKPOINT_PATH = "C:\path\to\checkpoint_epoch_10.pt"
   python -m uvicorn main:app --port 8000

   # Linux/macOS:
   export ROVF_CHECKPOINT_PATH="/path/to/checkpoint_epoch_10.pt"
   python -m uvicorn main:app --port 8000
   ```

When loaded, `/api/health` will report `"checkpointLoaded": true`.

---

## API Endpoints

- `GET /api/health` - Backend status, device (CPU/CUDA), checkpoint loaded status.
- `GET /api/gallery` - Returns reference gallery individuals with benchmark/sample status.
- `GET /api/experiments` - Real empirical study and citation metadata from IJCV 2026.
- `POST /api/identify` - Accepts `file` (MP4/MOV/WebM) or `clip_id`. Returns extracted frames, attention weights, and cosine similarity matches.

---

## Production Deployment

### Docker:
```bash
docker build -t wildtrace-backend .
docker run -p 8000:8000 wildtrace-backend
```

Compatible with Railway, Render, Fly.io, and Hugging Face Spaces.
