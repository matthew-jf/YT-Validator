# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "requests", "yt-dlp"]
# ///

import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yt_dlp

# Project Path Setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPER_BASE_URL = "https://forgeyt-video-mapper-backend-production.up.railway.app"

# Column Configurations
AUTO_YES_COLUMN = "AG_Action"
YOUTUBE_COLUMN = "(LINKED)    Video ID"

# Limits & Buffers
MAX_UPLOAD_BYTES = 90 * 1024 * 1024  # Strict 90MB cap to prevent Railway 413 errors
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 10 * 60
TARGET_SUCCESSFUL_DOWNLOADS = 10


def load_env_file(env_path=PROJECT_ROOT / ".env"):
    """Loads environment variables from a local .env file."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
            continue
        key, value = clean_line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_youtube_url(value):
    """Ensures raw IDs or partial strings are converted into a valid YouTube URL."""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith(("http://", "https://")):
        return text
    return f"https://www.youtube.com/watch?v={text}"


def download_video(url, download_dir):
    """
    Downloads the lowest resolution video available under 90MB.
    Uses OAuth2 authentication to bypass Chrome/Edge DPAPI encryption locks.
    """
    output_template = str(download_dir / "%(id)s.%(ext)s")
    
    ydl_opts = {
        # Force lowest video quality available to stay under the 90MB threshold
        "format": "worst[ext=mp4][filesize<90M]/worst[filesize<90M]/worst",
        "max_filesize": MAX_UPLOAD_BYTES,
        "outtmpl": output_template,
        "quiet": False,  # Changed to False so you can see the OAuth login URL in terminal
        "no_warnings": False,
        "noplaylist": True,
        
        # Bypasses local DPAPI errors by generating an interactive terminal login link
        "username": "oauth2", 
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    path = Path(filename)
    if not path.exists():
        matches = list(download_dir.glob(f"{info.get('id', '*')}.*"))
        if matches:
            path = matches[0]

    # Post-download safety buffer check
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        path.unlink(missing_ok=True)
        raise ValueError(f"Video is {path.stat().st_size} bytes. Exceeds Railway's 90MB safety ceiling. Skipping.")
    
    return path


def submit_match_job(video_path, token):
    """
    Submits raw binary payload to Railway. 
    Explicit Content-Length prevents Chunked Transfer Encoding SSLEOFErrors.
    """
    video_bytes = video_path.read_bytes()
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "video/mp4" if video_path.suffix.lower() == ".mp4" else "application/octet-stream",
        "Content-Length": str(len(video_bytes))
    }
    
    # Mount session retry mechanisms for dropped connections
    session = requests.Session()
    retries = Retry(
        total=3, 
        backoff_factor=2, 
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.post(
        f"{MAPPER_BASE_URL}/match-jobs",
        headers=headers,
        data=video_bytes,
        timeout=180, 
    )
    
    response.raise_for_status()
    return response.json()["jobId"]


def poll_match_job(job_id, token):
    """Polls the API until the job transitions out of queued/processing states."""
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = requests.get(
            f"{MAPPER_BASE_URL}/match-jobs/{job_id}",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Timed out waiting for mapper job {job_id}")


def empty_guess(status="", error=""):
    """Returns a blank schema structure for skipped or errored loops."""
    return {
        "Mapper_Status": status,
        "Mapper_Job_ID": "",
        "Guessed_Core_ID": "",
        "Guessed_Video_Variant_ID": "",
        "Guessed_Confidence": "",
        "Guessed_Match_Strength": "",
        "Mapper_Candidates_JSON": "",
        "Mapper_Error": error,
    }


def guess_from_payload(payload):
    """Parses top match candidate profiles directly from successful API payloads."""
    candidates = payload.get("candidates") or []
    top_candidate = candidates[0] if candidates else {}
    return {
        "Mapper_Status": payload.get("status", ""),
        "Mapper_Job_ID": payload.get("jobId", ""),
        "Guessed_Core_ID": top_candidate.get("coreId", ""),
        "Guessed_Video_Variant_ID": top_candidate.get("videoVariantId", ""),
        "Guessed_Confidence": top_candidate.get("confidence", ""),
        "Guessed_Match_Strength": top_candidate.get("matchStrength", ""),
        "Mapper_Candidates_JSON": json.dumps(candidates),
        "Mapper_Error": "",
    }


def build_auto_yes_mcid_guesses(
    data_path=PROJECT_ROOT / "data" / "output_claims.csv",
    output_path=PROJECT_ROOT / "data" / "auto_yes_mcid_guesses.csv",
):
    """Main pipeline execution loop."""
    load_env_file()
    token = os.environ.get("MAPPER_API_TOKEN")

    df = pd.read_csv(data_path)
    if AUTO_YES_COLUMN not in df.columns:
        raise KeyError(f"Missing required column: {AUTO_YES_COLUMN}")
    if YOUTUBE_COLUMN not in df.columns:
        raise KeyError(f"Missing required column: {YOUTUBE_COLUMN}")

    auto_yes_df = df[df[AUTO_YES_COLUMN].eq("Auto Yes")].copy()
    print(f"Preparing mapper guesses. Target total successes: {TARGET_SUCCESSFUL_DOWNLOADS}")

    for column in empty_guess().keys():
        auto_yes_df[column] = ""

    if not token:
        result = empty_guess(status="skipped", error="MAPPER_API_TOKEN is not set")
        for column, value in result.items():
            auto_yes_df[column] = value
        auto_yes_df.to_csv(output_path, index=False)
        print(f"MAPPER_API_TOKEN is not found. Saved blank placeholder rows to {output_path}")
        return

    success_count = 0

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        
        for row_number, row in auto_yes_df.iterrows():
            if success_count >= TARGET_SUCCESSFUL_DOWNLOADS:
                print(f"\n--- Processed {TARGET_SUCCESSFUL_DOWNLOADS} videos successfully! Stopping loop. ---")
                break

            url = normalize_youtube_url(row[YOUTUBE_COLUMN])
            if not url:
                result = empty_guess(status="skipped", error="Missing YouTube URL Target")
            else:
                try:
                    video_path = download_video(url, temp_dir)
                    job_id = submit_match_job(video_path, token)
                    payload = poll_match_job(job_id, token)
                    result = guess_from_payload(payload)
                    
                    success_count += 1
                    print(f"[{success_count}/{TARGET_SUCCESSFUL_DOWNLOADS}] Processing Complete: {url}")
                except Exception as exc:
                    print(f"Skipped URL {url} due to error: {exc}")
                    result = empty_guess(status="error", error=str(exc))

            for column, value in result.items():
                auto_yes_df.at[row_number, column] = value

    auto_yes_df.to_csv(output_path, index=False)
    print(f"\nFinalized MCID mapper file saved successfully to: {output_path}")


if __name__ == "__main__":
    build_auto_yes_mcid_guesses()