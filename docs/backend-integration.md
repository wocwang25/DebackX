# Backend Integration

Run MyDebackX as a separate HTTP worker. The web backend should call this worker over HTTP instead of importing OCR/MT code into the web process. In production, the web VM can stay small while this worker runs on a separate GPU host.

## Required Artifacts

- Worker code: `src/worker`, `src/pipeline`, `scripts`, `configs`, `Dockerfile.worker`.
- Runtime config: `configs/config-pipeline-strong.json`.
- Translation model volume: `models/mt-nllb-1p3b-en-vi/best`.
- Output volume: `outputs`.
- Optional offline OCR cache: PaddleOCR downloads `PP-OCRv5_server_det` and `en_PP-OCRv5_mobile_rec` on first run if they are not already cached.

Do not commit `models`, `outputs`, or dataset folders to the web backend repository.

## Self-Contained Release Image

Use `Dockerfile.release` when you want one image that already contains the trained production checkpoint and pre-downloaded PaddleOCR models. This image does not require training or a model volume at runtime.

It includes:

- `models/mt-nllb-1p3b-en-vi/best`
- `configs/config-pipeline-strong.json`
- `src`, `scripts`, worker API
- PaddleOCR PP-OCRv5 detection and recognition models downloaded during build
- DejaVu font for Vietnamese rendering

It intentionally does not include:

- training dataset
- old checkpoints such as `last`, TrOCR, or the smaller legacy MT model
- previous outputs/jobs

Build locally:

```bash
sh scripts/prepare-release-assets.sh
IMAGE=iimt-worker:release sh scripts/build-release-image.sh
```

Run locally:

```bash
docker run --gpus all --rm -p 8000:8000 iimt-worker:release
```

Run with persistent outputs:

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v "$(pwd)/outputs:/app/outputs" \
  iimt-worker:release
```

Push to Docker Hub:

```bash
docker login -u <dockerhub-user>

sh scripts/prepare-release-assets.sh

DOCKERHUB_USER=<dockerhub-user> \
IMAGE_NAME=iimt-worker \
TAG=release \
PUSH=1 \
sh scripts/build-release-image.sh
```

If you use `sudo docker`, login and build with the same Docker user:

```bash
sudo docker login -u <dockerhub-user>

sh scripts/prepare-release-assets.sh

sudo env DOCKERHUB_USER=<dockerhub-user> \
  IMAGE_NAME=iimt-worker \
  TAG=release \
  PUSH=1 \
  sh scripts/build-release-image.sh
```

Use a private Docker Hub repository if the checkpoint should not be public. This release image contains the trained model weights.

On another GPU host:

```bash
docker pull <dockerhub-user>/iimt-worker:release
docker run --gpus all --rm -p 8000:8000 \
  -e IIMT_WORKER_API_KEY="<shared-secret>" \
  <dockerhub-user>/iimt-worker:release
```

Set the same value in the VieTrans gateway:

```env
IIMT_WORKER_URL=http://<gpu-host>:8000
IIMT_WORKER_MODE=async
IIMT_WORKER_API_KEY=<shared-secret>
```

## Docker Run

Use `Dockerfile.worker` for development or when you want to keep models outside the image and mount them at runtime.

```bash
docker build -f Dockerfile.worker -t iimt-worker .

docker run --gpus all --rm -p 8000:8000 \
  -e CONFIG=configs/config-pipeline-strong.json \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/outputs:/app/outputs" \
  iimt-worker
```

## GPU Runtime Setup

If Docker fails with:

```text
failed to discover GPU vendor from CDI: no known GPU vendor found
```

the host NVIDIA driver can see the GPU, but Docker cannot expose it to containers. Install and configure the NVIDIA Container Toolkit on the host:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Validate GPU access before running the worker:

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If the host still reports CDI errors after installing the toolkit, regenerate the CDI spec:

```bash
sudo systemctl enable --now nvidia-cdi-refresh.path
sudo systemctl restart nvidia-cdi-refresh.service
nvidia-ctk cdi list
```

For Docker Compose, the web backend should use an internal URL such as:

```text
IIMT_WORKER_URL=http://iimt-worker:8000
```

If the worker is on another machine, use its private or public API URL instead:

```text
IIMT_WORKER_URL=https://debackx-worker.example.com
```

## Health Check

```bash
curl http://localhost:8000/health
```

The response should include:

```text
status: ok
mt_checkpoint_exists: true
models_loaded: true
```

If jobs fail with `OSError: cannot open resource`, the worker image is missing the configured render font. Rebuild the image after installing `fonts-dejavu-core`, or set `render.font_path` in `configs/config-pipeline-strong.json` to a font that exists inside the container.

## Optional Worker API Key

When `IIMT_WORKER_API_KEY` is set, the worker requires the key on `/jobs`, `/jobs/{job_id}`, `/translate`, and `/files/...`. Send either header:

```text
Authorization: Bearer <shared-secret>
X-API-Key: <shared-secret>
```

`/health` stays public by default so Docker and load balancer health checks keep working. To protect it too, set:

```env
IIMT_WORKER_PROTECT_HEALTH=true
```

## API Contract

Create an async translation job:

```bash
curl -X POST "$IIMT_WORKER_URL/jobs" \
  -H "Authorization: Bearer $IIMT_WORKER_API_KEY" \
  -F "file=@english-image.jpg"
```

Response:

```json
{
  "job_id": "abc123",
  "status": "queued",
  "status_url": "/jobs/abc123",
  "input_url": "/files/uploads/abc123.jpg"
}
```

Poll the job:

```bash
curl "$IIMT_WORKER_URL/jobs/abc123" \
  -H "Authorization: Bearer $IIMT_WORKER_API_KEY"
```

When `status` is `succeeded`, use:

```text
result.output_url
result.mask_url
result.metadata_url
result.regions
```

The URLs are relative to the worker base URL. For example, combine `IIMT_WORKER_URL + result.output_url`.
If `IIMT_WORKER_API_KEY` is set, include the same key when downloading those URLs.

Use `/translate` only for demos or small synchronous requests. Production backend code should prefer `/jobs` plus polling.

## Node Backend Example

```js
const workerUrl = process.env.IIMT_WORKER_URL;
const workerHeaders = process.env.IIMT_WORKER_API_KEY
  ? { Authorization: `Bearer ${process.env.IIMT_WORKER_API_KEY}` }
  : {};

async function createTranslationJob(file) {
  const form = new FormData();
  form.append("file", new Blob([file.buffer], { type: file.mimetype }), file.originalname);

  const response = await fetch(`${workerUrl}/jobs`, {
    method: "POST",
    headers: workerHeaders,
    body: form,
  });

  if (!response.ok) {
    throw new Error(`IIMT worker failed: ${response.status} ${await response.text()}`);
  }

  return response.json();
}

async function getTranslationJob(jobId) {
  const response = await fetch(`${workerUrl}/jobs/${jobId}`, {
    headers: workerHeaders,
  });
  if (!response.ok) {
    throw new Error(`IIMT worker failed: ${response.status} ${await response.text()}`);
  }
  return response.json();
}
```

## Python Backend Example

```python
import os
import requests


def worker_headers():
    api_key = os.environ.get("IIMT_WORKER_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def create_translation_job(worker_url, image_path):
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{worker_url}/jobs",
            files={"file": (image_path.name, image_file, "image/jpeg")},
            headers=worker_headers(),
            timeout=30,
        )
    response.raise_for_status()
    return response.json()


def get_translation_job(worker_url, job_id):
    response = requests.get(
        f"{worker_url}/jobs/{job_id}",
        headers=worker_headers(),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
```
