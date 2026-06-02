import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles


ROOT_DIR = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT_DIR / "src" / "pipeline"
sys.path.append(str(PIPELINE_DIR))

from translate_real_image import RealImageTranslator


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json_config(path):
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file), config_path


def resolve_from_config(config_path, raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


CONFIG_PATH = os.environ.get("IIMT_CONFIG", "configs/config-pipeline.json")
CONFIG, CONFIG_FILE = load_json_config(CONFIG_PATH)
WORKER_CONFIG = CONFIG.get("worker", {})
OUTPUT_DIR = resolve_from_config(CONFIG_FILE, WORKER_CONFIG.get("output_dir", "../outputs/worker"))
JOBS_DIR = OUTPUT_DIR / "jobs"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
RESULTS_DIR = OUTPUT_DIR / "results"
ALLOWED_EXTENSIONS = set(WORKER_CONFIG.get("allowed_extensions", [".jpg", ".jpeg", ".png", ".webp"]))
MAX_UPLOAD_BYTES = int(WORKER_CONFIG.get("max_upload_mb", 20)) * 1024 * 1024

for directory in [OUTPUT_DIR, JOBS_DIR, UPLOADS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="IIMT English-Vietnamese Worker", version="1.0.0")
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")

translator = None
translator_lock = threading.Lock()
jobs_lock = threading.Lock()


def get_translator():
    global translator
    if translator is None:
        translator = RealImageTranslator(CONFIG_FILE, lazy=True)
    return translator


def output_url(path):
    path = Path(path).resolve()
    rel_path = path.relative_to(OUTPUT_DIR.resolve())
    return "/files/" + quote(rel_path.as_posix())


def job_file(job_id):
    return JOBS_DIR / f"{job_id}.json"


def write_job(job):
    job["updated_at"] = utc_now()
    with jobs_lock:
        with job_file(job["job_id"]).open("w", encoding="utf-8") as job_json:
            json.dump(job, job_json, ensure_ascii=False, indent=2)


def read_job(job_id):
    path = job_file(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    with path.open("r", encoding="utf-8") as job_json:
        return json.load(job_json)


def new_job_record(job_id, input_path, mode):
    now = utc_now()
    return {
        "job_id": job_id,
        "status": "queued",
        "mode": mode,
        "created_at": now,
        "updated_at": now,
        "input_path": str(input_path),
        "input_url": output_url(input_path),
        "result": None,
        "error": None,
    }


def validate_extension(filename):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"unsupported file extension. Allowed: {allowed}")
    return suffix


async def save_upload(file, job_id):
    suffix = validate_extension(file.filename)
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"file too large. Max upload is {MAX_UPLOAD_BYTES} bytes")
    upload_path = UPLOADS_DIR / f"{job_id}{suffix}"
    upload_path.write_bytes(content)
    return upload_path


def run_translation_job(job_id, input_path):
    job = read_job(job_id)
    job["status"] = "running"
    job["started_at"] = utc_now()
    write_job(job)

    output_path = RESULTS_DIR / job_id / "translated.png"
    metadata_path = RESULTS_DIR / job_id / "metadata.json"
    try:
        with translator_lock:
            service = get_translator()
            result = service.process_image(input_path, output=output_path, metadata=metadata_path)

        result_payload = {
            "output_path": result["output"],
            "output_url": output_url(result["output"]),
            "mask_path": result["mask"],
            "mask_url": output_url(result["mask"]),
            "metadata_path": result["metadata"],
            "metadata_url": output_url(result["metadata"]),
            "num_regions": result["num_regions"],
            "regions": result["regions"],
        }
        job["status"] = "succeeded"
        job["finished_at"] = utc_now()
        job["result"] = result_payload
        job["error"] = None
    except Exception as exc:
        job["status"] = "failed"
        job["finished_at"] = utc_now()
        job["error"] = {"type": exc.__class__.__name__, "message": str(exc)}
    write_job(job)
    return job


@app.on_event("startup")
def startup():
    service = get_translator()
    if bool(WORKER_CONFIG.get("load_models_on_startup", True)):
        service.load()


@app.get("/health")
def health():
    service = get_translator()
    payload = service.health()
    payload.update(
        {
            "status": "ok",
            "config": str(CONFIG_FILE),
            "worker_output_dir": str(OUTPUT_DIR),
            "max_upload_mb": WORKER_CONFIG.get("max_upload_mb", 20),
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        }
    )
    return payload


@app.post("/jobs", status_code=202)
async def create_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    input_path = await save_upload(file, job_id)
    job = new_job_record(job_id, input_path, mode="async")
    write_job(job)
    background_tasks.add_task(run_translation_job, job_id, input_path)
    return {
        "job_id": job_id,
        "status": job["status"],
        "status_url": f"/jobs/{job_id}",
        "input_url": job["input_url"],
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return read_job(job_id)


@app.post("/translate")
async def translate_now(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex
    input_path = await save_upload(file, job_id)
    job = new_job_record(job_id, input_path, mode="sync")
    write_job(job)
    run_translation_job(job_id, input_path)
    return read_job(job_id)
