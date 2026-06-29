# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "requests", "yt-dlp", "openai-whisper", "torch"]
# ///

import json
import os
import tempfile
import time
import re
import math
import subprocess
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yt_dlp
import whisper

try:
    import torch
except ImportError:
    torch = None

# ==========================================
# 1. PIPELINE CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAPPER_BASE_URL = "https://forgeyt-video-mapper-backend-production.up.railway.app"

# Columns
AUTO_YES_COLUMN = "AG_Action"
YOUTUBE_COLUMN = "(LINKED)    Video ID"
EXPECTED_LANGUAGE_COLUMN = "Expected Language"
WESS_LANGUAGE_ID_COLUMN = "Actual WESS Language ID"
VIDEO_TITLE_COLUMN = "Video Title"

# Limits & Buffers
MAX_UPLOAD_BYTES = 90 * 1024 * 1024  
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 10 * 60  
TARGET_SUCCESSFUL_DOWNLOADS = 5

# Whisper Config
WHISPER_MODEL_NAME = "small"
CLIP_SECONDS = 10
SAMPLE_CONFIGS = [
    {"sample_number": 1, "percent": 50, "mode": "center"},
    {"sample_number": 2, "percent": 30, "mode": "center"},
    {"sample_number": 3, "percent": 3, "mode": "start"},
]
HIGH_CONFIDENCE = 0.85
DOMINANT_CONFIDENCE = 0.90
MIN_AGREEMENT_CONFIDENCE = 0.70
MIN_DOMINANCE_MARGIN = 0.10


# ==========================================
# 2. LANGUAGE DICTIONARIES & HELPERS
# ==========================================
HIGH_PRIORITY_LANGUAGES = [
    "English", "Hindi", "Spanish", "French", "Indonesian", "Amharic", "Tamil", "Vietnamese",
    "Russian", "Portuguese", "Swahili", "Malay", "Telugu", "Tagalog", "Armenian", "Kannada",
    "Urdu", "Haitian Creole", "Romanian", "Greek", "Arabic", "Burmese", "Rejang", "Tobelo",
    "Loloda", "Galela", "Mentawai", "Gayo", "Aceh", "Chichewa", "Kinyarwanda",
]

EQUIVALENT_LANGUAGES = {
    "Spanish Latin American": "Spanish", "Spanish, Latin American": "Spanish",
    "Latin American Spanish": "Spanish", "Spanish Castilian": "Spanish",
    "Spanish, Castilian": "Spanish", "Castilian Spanish": "Spanish",
    "Portuguese Brazil": "Portuguese", "Portuguese, Brazil": "Portuguese",
    "Brazilian Portuguese": "Portuguese", "Arabic Modern Standard": "Arabic",
    "Arabic, Modern Std": "Arabic", "Modern Standard Arabic": "Arabic",
    "MSA Arabic": "Arabic", "Indonesian Yesus": "Indonesian",
    "Indonesian (Yesus)": "Indonesian", "Bahasa Indonesia": "Indonesian",
    "Swahili Kenya": "Swahili", "Swahili, Kenya": "Swahili",
    "Swahili: Kenya": "Swahili", "Kenyan Swahili": "Swahili",
    "Kiswahili": "Swahili", "Burmese Standard": "Burmese",
    "Burmese, Standard": "Burmese", "Creole French Haitian": "Haitian Creole",
    "Creole French, Haitian": "Haitian Creole", "Creole, Haitian": "Haitian Creole",
    "Haitian Creole French": "Haitian Creole", "Tigrinya (Eritrea)": "Tigrinya",
    "Chinese, Simplified": "Chinese", "Chinese, Mandarin": "Chinese",
    "Chinese, Traditional": "Chinese", "Galela": "Galela", "Galala": "Galela",
    "Farsi, Western": "Farsi, Western", "Farsi": "Farsi, Western",
    "Persian": "Farsi, Western", "Mbandja": "Mandja", "Mandja": "Mbandja",
}

WHISPER_CODE_TO_LANGUAGE = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian",
    "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali",
    "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer",
    "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan",
    "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati",
    "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", "mt": "Maltese",
    "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Burmese", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala",
    "ha": "Hausa", "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese",
}

def canonical_language(value):
    value = "" if value is None else str(value).strip()
    return EQUIVALENT_LANGUAGES.get(value, value)

def normalize_title(text):
    text = "" if text is None else str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def build_language_lookup():
    lookup = {}
    def add_alias(alias, canonical, priority=False):
        alias = normalize_title(alias)
        if not alias or (not priority and len(alias) < 5): return
        lookup[alias] = canonical_language(canonical)
        
    for lang in HIGH_PRIORITY_LANGUAGES:
        add_alias(lang, lang, priority=True)
    manual_aliases = {
        "Latin American Spanish": "Spanish", "Castilian Spanish": "Spanish", "Brazilian Portuguese": "Portuguese",
        "Modern Standard Arabic": "Arabic", "MSA Arabic": "Arabic", "Bahasa Indonesia": "Indonesian",
        "Kiswahili": "Swahili", "Espanol": "Spanish", "Español": "Spanish",
        "Portugues": "Portuguese", "Português": "Portuguese", "Kreyol": "Haitian Creole",
    }
    for alias, canonical in manual_aliases.items():
        add_alias(alias, canonical, priority=True)
        
    return sorted(lookup.items(), key=lambda x: len(x[0]), reverse=True)

LANGUAGE_LOOKUP = build_language_lookup()

def detect_language_from_title(title):
    padded = f" {normalize_title(title)} "
    for alias, canonical in LANGUAGE_LOOKUP:
        if f" {alias} " in padded:
            return canonical, alias
    return "", ""


# ==========================================
# 3. WHISPER LANGUAGE CHECKER CORE
# ==========================================
def load_whisper_model():
    cuda_available = bool(torch and torch.cuda.is_available())
    device = "cuda" if cuda_available else "cpu"
    print(f"[*] Loading OpenAI Whisper '{WHISPER_MODEL_NAME}' on {device}...")
    return whisper.load_model(WHISPER_MODEL_NAME, device=device)

def calculate_sample_starts(duration_seconds):
    starts = {}
    for config in SAMPLE_CONFIGS:
        percent = float(config["percent"])
        if config.get("mode") == "start":
            start = duration_seconds * percent / 100.0
        else:
            start = (duration_seconds * percent / 100.0) - (CLIP_SECONDS / 2.0)
        max_start = max(duration_seconds - CLIP_SECONDS, 0.0)
        starts[config["sample_number"]] = round(min(max(float(start), 0.0), max_start), 3)
    return starts

def extract_local_clip(video_path, start_seconds, workdir):
    clip_path = Path(workdir) / f"clip_{start_seconds}.wav"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_seconds), "-t", str(CLIP_SECONDS), "-i", str(video_path),
        "-ac", "1", "-ar", "16000", str(clip_path),
    ]
    subprocess.run(cmd, check=True, timeout=60)
    return clip_path

def detect_clip_language(model, clip_path):
    audio = whisper.load_audio(str(clip_path))
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probabilities = model.detect_language(mel)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    
    return {
        "language_name": canonical_language(WHISPER_CODE_TO_LANGUAGE.get(ranked[0][0], ranked[0][0])),
        "confidence": float(ranked[0][1]),
        "second_language_name": canonical_language(WHISPER_CODE_TO_LANGUAGE.get(ranked[1][0], ranked[1][0])),
        "second_confidence": float(ranked[1][1]),
        "confidence_margin": float(ranked[0][1]) - float(ranked[1][1]),
    }

def vote_whisper_samples(samples):
    groups = {}
    for s in samples:
        lang = canonical_language(s["language_name"])
        groups.setdefault(lang, []).append(s)

    english = groups.get("English", [])
    strong_english = [s for s in english if s["confidence"] >= HIGH_CONFIDENCE]
    if len(strong_english) >= 2:
        return {"accepted": True, "language": "English", "reason": "Accepted: English confirmed by multiple independent samples."}

    if len(groups) == 1:
        language, agreeing = next(iter(groups.items()))
        if language != "English":
            avg_conf = sum(s["confidence"] for s in agreeing) / len(agreeing)
            if any(s["confidence"] >= HIGH_CONFIDENCE for s in agreeing) or avg_conf >= 0.75:
                return {"accepted": True, "language": language, "reason": "Accepted: 3-of-3 Whisper consensus."}
        return {"accepted": False, "language": "UNKNOWN", "reason": "Human Review: 3-of-3 consensus did not meet requirements."}

    for language, agreeing in groups.items():
        if language == "English" or len(agreeing) != 2: continue
        confs = [s["confidence"] for s in agreeing]
        if all(c >= MIN_AGREEMENT_CONFIDENCE for c in confs) and any(c >= HIGH_CONFIDENCE for c in confs):
            return {"accepted": True, "language": language, "reason": "Accepted: 2-of-3 Whisper consensus."}

    if len(groups) == 3:
        top = max(samples, key=lambda s: s["confidence"])
        margin = min(top["confidence"] - s["confidence"] for s in samples if s is not top)
        if top["language_name"] != "English" and top["confidence"] >= DOMINANT_CONFIDENCE and margin >= MIN_DOMINANCE_MARGIN and top["confidence_margin"] >= MIN_DOMINANCE_MARGIN:
            return {"accepted": True, "language": canonical_language(top["language_name"]), "reason": "Accepted: dominant non-English Whisper prediction."}

    return {"accepted": False, "language": "UNKNOWN", "reason": "Human Review: no language has sufficient combined evidence."}

def run_language_checker(video_path, duration_seconds, expected_language, title, whisper_model, workdir):
    print(f"  -> [Step 4] Running Language Checker...")
    expected_language = canonical_language(expected_language)
    
    # 1. Title Check
    title_lang, alias = detect_language_from_title(title)
    if title_lang:
        print(f"     Title matched alias '{alias}' -> {title_lang}")
        return {
            "Lang_Predicted": title_lang,
            "Lang_Is_Match": True if not expected_language else canonical_language(title_lang) == expected_language,
            "Lang_Method": "TITLE",
            "Lang_Verdict": "TITLE_LANGUAGE_FOUND",
            "Lang_Reason": f"Matched title alias: {alias}"
        }

    if not duration_seconds or duration_seconds < (CLIP_SECONDS * 2):
        return {"Lang_Predicted": "UNKNOWN", "Lang_Verdict": "HUMAN_REVIEW", "Lang_Reason": "Video duration too short or missing."}

    # 2. Whisper Check
    print(f"     Extracting audio clips via ffmpeg & running Whisper...")
    starts = calculate_sample_starts(duration_seconds)
    samples = []
    
    for _, start in starts.items():
        clip_path = extract_local_clip(video_path, start, workdir)
        evidence = detect_clip_language(whisper_model, clip_path)
        samples.append(evidence)
        clip_path.unlink(missing_ok=True)
        print(f"       - Sample @ {start}s: {evidence['language_name']} ({evidence['confidence']:.2f})")

    decision = vote_whisper_samples(samples)
    print(f"     Whisper Result: {decision['language']} | Reason: {decision['reason']}")
    
    return {
        "Lang_Predicted": decision["language"],
        "Lang_Is_Match": True if not expected_language else decision["language"] == expected_language,
        "Lang_Method": "WHISPER_VOTE" if decision["accepted"] else "HUMAN_REVIEW",
        "Lang_Verdict": "WHISPER_LANGUAGE_FOUND" if decision["accepted"] else "HUMAN_REVIEW",
        "Lang_Reason": decision["reason"]
    }

def empty_language_guess():
    return {
        "Lang_Predicted": "",
        "Lang_Is_Match": "",
        "Lang_Method": "",
        "Lang_Verdict": "",
        "Lang_Reason": ""
    }


# ==========================================
# 4. MAPPER CORE (Network/API)
# ==========================================
def load_env_file(env_path=PROJECT_ROOT / ".env"):
    if not env_path.exists(): return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if not clean_line or clean_line.startswith("#") or "=" not in clean_line: continue
        key, value = clean_line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def normalize_youtube_url(value):
    text = str(value).strip()
    if not text or text.lower() == "nan": return None
    if text.startswith(("http://", "https://")): return text
    return f"https://www.youtube.com/watch?v={text}"

def download_video(url, download_dir):
    """
    Downloads the absolute bare minimum required for a matching algorithm.
    Forces 144p resolution and only downloads the first 3 minutes.
    """
    output_template = str(download_dir / "%(id)s.%(ext)s")
    
    ydl_opts = {
        "format": "worstvideo[height<=144][ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst",
        "max_filesize": MAX_UPLOAD_BYTES,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "download_ranges": lambda info_dict, ydl: [{"start_time": 0, "end_time": 180}],
        "force_keyframes_at_cuts": True,
    }

    print(f"  -> [Step 1] Downloading first 3 mins at minimum resolution (144p)...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # Handle duration safely if yt-dlp couldn't fetch it
        duration = info.get("duration", 0) 
        if duration > 180:
            duration = 180  # Cap the duration for the Language Checker to 3 minutes

    path = Path(filename)
    if not path.exists():
        matches = list(download_dir.glob(f"{info.get('id', '*')}.*"))
        if matches:
            path = matches[0]

    # Post-download safety buffer check
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  -> Download complete! File size: {file_size_mb:.2f} MB")
    
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        path.unlink(missing_ok=True)
        raise ValueError(f"Video is {path.stat().st_size} bytes. Exceeds Railway's 90MB safety ceiling. Skipping.")
    
    return path, duration

def submit_match_job(video_path, token):
    print(f"  -> [Step 2] Uploading binary to Railway API...")
    video_bytes = video_path.read_bytes()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "video/mp4" if video_path.suffix.lower() == ".mp4" else "application/octet-stream",
        "Content-Length": str(len(video_bytes))
    }
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504], allowed_methods=["POST"])))

    response = session.post(f"{MAPPER_BASE_URL}/match-jobs", headers=headers, data=video_bytes, timeout=180)
    response.raise_for_status()
    return response.json()["jobId"]

def poll_match_job(job_id, token):
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    print(f"  -> [Step 3] Polling Railway for match results...", end="\n", flush=True)
    last_status = None
    
    while time.monotonic() < deadline:
        response = requests.get(f"{MAPPER_BASE_URL}/match-jobs/{job_id}", headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        
        if status in {"succeeded", "failed", "cancelled"}:
            print(f"     -> Job finished with status: {status.upper()}")
            return payload
            
        if status != last_status:
            print(f"     Status: {status.upper()} (Waiting...)")
            last_status = status
            
        time.sleep(POLL_INTERVAL_SECONDS)
        
    raise TimeoutError(f"Timed out waiting for mapper job {job_id}")

def empty_guess(status="", error=""):
    return {
        "Mapper_Status": status, "Mapper_Job_ID": "", "Guessed_Core_ID": "",
        "Guessed_Video_Variant_ID": "", "Guessed_Confidence": "", 
        "Guessed_Match_Strength": "", "Mapper_Error": error,
    }

def guess_from_payload(payload):
    candidates = payload.get("candidates") or []
    top = candidates[0] if candidates else {}
    return {
        "Mapper_Status": payload.get("status", ""), "Mapper_Job_ID": payload.get("jobId", ""),
        "Guessed_Core_ID": top.get("coreId", ""), "Guessed_Video_Variant_ID": top.get("videoVariantId", ""),
        "Guessed_Confidence": top.get("confidence", ""), "Guessed_Match_Strength": top.get("matchStrength", ""),
        "Mapper_Error": "",
    }


# ==========================================
# 5. MAIN PIPELINE LOOP
# ==========================================
def build_auto_yes_mcid_guesses(
    data_path=PROJECT_ROOT / "data" / "output_claims.csv",
    output_path=PROJECT_ROOT / "data" / "auto_yes_mcid_guesses_unified.csv",
):
    print("=====================================================")
    print("      STARTING UNIFIED MAPPER & LANGUAGE PIPELINE    ")
    print("=====================================================")
    
    load_env_file()
    token = os.environ.get("MAPPER_API_TOKEN")

    # 1. Init Data
    df = pd.read_csv(data_path)
    if AUTO_YES_COLUMN not in df.columns or YOUTUBE_COLUMN not in df.columns:
        raise KeyError("Missing required columns in CSV")

    auto_yes_df = df[df[AUTO_YES_COLUMN].eq("Auto Yes")].copy()
    print(f"[*] Found {len(auto_yes_df)} 'Auto Yes' rows.")

    # Apply empty schema structures to dataframe
    for col in empty_guess().keys(): auto_yes_df[col] = ""
    for col in empty_language_guess().keys(): auto_yes_df[col] = ""

    if not token:
        print("[!] MAPPER_API_TOKEN missing. Aborting.")
        return

    # 2. Init AI Models
    whisper_model = load_whisper_model()
    success_count = 0

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        
        for row_number, row in auto_yes_df.iterrows():
            if success_count >= TARGET_SUCCESSFUL_DOWNLOADS: break

            url = normalize_youtube_url(row[YOUTUBE_COLUMN])
            title = str(row.get(VIDEO_TITLE_COLUMN, ""))
            expected_lang = str(row.get(EXPECTED_LANGUAGE_COLUMN, ""))
            # Fallback to WESS ID if expected text is blank
            if not expected_lang and WESS_LANGUAGE_ID_COLUMN in row:
                expected_lang = str(row[WESS_LANGUAGE_ID_COLUMN])

            print(f"\n=====================================================")
            print(f"Attempting to process: {url}")
            
            if not url:
                result = {**empty_guess("skipped", "Missing YouTube URL"), **empty_language_guess()}
            else:
                try:
                    # 1. Download
                    video_path, duration = download_video(url, temp_dir)
                    
                    # 2. MCID Mapper
                    job_id = submit_match_job(video_path, token)
                    payload = poll_match_job(job_id, token)
                    mapper_result = guess_from_payload(payload)
                    
                    # 3. Language Checker (using local file)
                    lang_result = run_language_checker(
                        video_path=video_path,
                        duration_seconds=duration,
                        expected_language=expected_lang,
                        title=title,
                        whisper_model=whisper_model,
                        workdir=temp_dir
                    )
                    
                    result = {**mapper_result, **lang_result}
                    success_count += 1
                    print(f"[*] [{success_count}/{TARGET_SUCCESSFUL_DOWNLOADS}] SUCCESS! Finished processing {url}")
                    
                except Exception as exc:
                    print(f"  -> [ERROR] Skipped URL {url} due to error: {exc}")
                    result = {**empty_guess("error", str(exc)), **empty_language_guess()}

            # Write data to row
            for column, value in result.items():
                auto_yes_df.at[row_number, column] = value

    print(f"\n[*] Saving finalized unified file...")
    auto_yes_df.to_csv(output_path, index=False)
    print(f"[*] DONE! File saved successfully to: {output_path}")

if __name__ == "__main__":
    build_auto_yes_mcid_guesses()