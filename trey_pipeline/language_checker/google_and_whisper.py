# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "requests", "yt-dlp", "openai-whisper", "torch"]
# ///

import json
import os
import tempfile
import time
import re
import subprocess
import html
from pathlib import Path

import pandas as pd
import requests
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

# Columns
AUTO_YES_COLUMN = "AG_Action" 
YOUTUBE_COLUMN = "video_id" 
EXPECTED_LANGUAGE_COLUMN = "Expected Language"
WESS_LANGUAGE_ID_COLUMN = "language_id" 
VIDEO_TITLE_COLUMN = "video_title"

# Limits & Buffers
MAX_DOWNLOAD_BYTES = 90 * 1024 * 1024  

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

def load_wess_mapping(csv_path=PROJECT_ROOT / "data" / "WESS ID.csv"):
    try:
        # Load the CSV
        wess_df = pd.read_csv(csv_path)
        # Strip headers of hidden spaces
        wess_df.columns = wess_df.columns.astype(str).str.strip()
        
        name_col = [c for c in wess_df.columns if "Language Name" in c][0]
        id_col = [c for c in wess_df.columns if "WESS Language ID" in c][0]
        
        wess_dict = {}
        for _, row in wess_df.iterrows():
            raw_id = str(row[id_col]).split('.')[0].strip()
            raw_name = str(row[name_col]).strip()
            if raw_id and raw_id.lower() != 'nan':
                wess_dict[raw_id] = raw_name
        return wess_dict
    except Exception as e:
        print(f"[!] Warning: {csv_path} not found or error reading. WESS ID translation will fail. {e}")
        return {}

# Build the dictionary when the script starts
WESS_ID_TO_LANGUAGE = load_wess_mapping()

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
        # If priority=False, we require the language name to be at least 4 letters long 
        # to avoid accidentally triggering on random short words in the title.
        if not alias or (not priority and len(alias) < 4): return
        lookup[alias] = canonical_language(canonical)
        
    # 1. Add our hardcoded high-priority list
    for lang in HIGH_PRIORITY_LANGUAGES:
        add_alias(lang, lang, priority=True)
        
    # 2. Add manual alias rules (MSA Arabic, Castilian, etc.)
    manual_aliases = {
        "Latin American Spanish": "Spanish", "Castilian Spanish": "Spanish", "Brazilian Portuguese": "Portuguese",
        "Modern Standard Arabic": "Arabic", "MSA Arabic": "Arabic", "Bahasa Indonesia": "Indonesian",
        "Kiswahili": "Swahili", "Espanol": "Spanish", "Español": "Spanish",
        "Portugues": "Portuguese", "Português": "Portuguese", "Kreyol": "Haitian Creole",
    }
    for alias, canonical in manual_aliases.items():
        add_alias(alias, canonical, priority=True)
        
    # 3. Dynamically absorb every language name from the WESS ID.csv
    for wess_lang_name in WESS_ID_TO_LANGUAGE.values():
        # Some WESS names look like "Yao (Iu Mien)". We add it as-is, and our normalize_title 
        # function will automatically strip out the punctuation so it matches clean text.
        add_alias(wess_lang_name, wess_lang_name, priority=False)
        
    # Sort by length descending, so we check for long compound names before short names
    return sorted(lookup.items(), key=lambda x: len(x[0]), reverse=True)

# Build the lookup immediately after defining it
LANGUAGE_LOOKUP = build_language_lookup()

def detect_language_from_title(title):
    padded = f" {normalize_title(title)} "
    for alias, canonical in LANGUAGE_LOOKUP:
        if f" {alias} " in padded:
            return canonical, alias
    return "", ""


# ==========================================
# 3. GOOGLE TRANSLATE API HELPER
# ==========================================
def translate_text(text, target_lang="en"):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "[Translation Error: GOOGLE_API_KEY missing from environment]", ""
    if not text or not text.strip():
        return "", ""

    try:
        url = "https://translation.googleapis.com/language/translate/v2"
        response = requests.post(
            url,
            params={"key": api_key},
            json={"q": [text], "target": target_lang},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        translation_data = data["data"]["translations"][0]
        clean_text = html.unescape(translation_data["translatedText"])
        detected_lang = translation_data.get("detectedSourceLanguage", "")
        
        return clean_text, detected_lang
    except Exception as e:
        return f"[Translation Error: {e}]", ""


# ==========================================
# 4. WHISPER AUDIO HANDLING
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
    
    transcription = model.transcribe(str(clip_path))
    whisper_text = transcription.get("text", "").strip()
    
    gt_text, gt_lang = "", ""
    if whisper_text:
        gt_text, gt_lang = translate_text(whisper_text, target_lang="en")
    
    return {
        "language_name": canonical_language(WHISPER_CODE_TO_LANGUAGE.get(ranked[0][0], ranked[0][0])),
        "confidence": float(ranked[0][1]),
        "second_language_name": canonical_language(WHISPER_CODE_TO_LANGUAGE.get(ranked[1][0], ranked[1][0])),
        "second_confidence": float(ranked[1][1]),
        "confidence_margin": float(ranked[0][1]) - float(ranked[1][1]),
        "whisper_text": whisper_text,
        "gt_text": gt_text,
        "gt_lang": gt_lang
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


# ==========================================
# 5. CORE ENGINE (Whisper vs Google Translate)
# ==========================================
def run_language_checker(video_path, duration_seconds, expected_language, title, whisper_model, workdir):
    print(f"  -> Evaluating Video Language...")
    expected_language = canonical_language(expected_language)
    
    title_lang, alias = detect_language_from_title(title)
    if title_lang:
        print(f"     [Title Rule Match] Alias: '{alias}' -> Predicted: {title_lang}")
        res = empty_language_guess()
        res.update({
            "Lang_Predicted": title_lang,
            "Lang_Is_Match": str(True if not expected_language else canonical_language(title_lang) == expected_language),
            "Lang_Method": "TITLE",
            "Lang_Verdict": "TITLE_LANGUAGE_FOUND",
            "Lang_Reason": f"Matched local title alias: {alias}"
        })
        return res

    if not duration_seconds or duration_seconds < (CLIP_SECONDS * 2):
        res = empty_language_guess()
        res.update({"Lang_Predicted": "UNKNOWN", "Lang_Verdict": "HUMAN_REVIEW", "Lang_Reason": "Video duration too short/missing."})
        return res

    print(f"     Title unknown. Initiating Whisper vs. Google Cloud Translate Head-to-Head...")
    
    # 1. Base Title Translation via Google Cloud Translate
    gt_title_lang, gt_title_translation = "", ""
    if title.strip():
        gt_title_translation, gt_title_lang = translate_text(title, target_lang="en")
        if gt_title_translation and not gt_title_translation.startswith("[Translation Error"):
            print(f"     [GT Title Analysis] Detected: {gt_title_lang.upper()} | Translated: {gt_title_translation}")

    # 2. Extract Audio Clips and pit Whisper against Google Translate side-by-side
    starts = calculate_sample_starts(duration_seconds)
    samples = []
    
    for _, start in starts.items():
        clip_path = extract_local_clip(video_path, start, workdir)
        evidence = detect_clip_language(whisper_model, clip_path)
        samples.append(evidence)
        clip_path.unlink(missing_ok=True)
        
        # Terminal Side-by-Side Comparison Layout
        print(f"       --- Audio Segment @ {start}s ---")
        print(f"         Whisper Language Tag    : {evidence['language_name']} ({evidence['confidence']:.2f})")
        print(f"         Whisper Transcript      : \"{evidence['whisper_text']}\"")
        print(f"         Google Translate (Audio): \"{evidence['gt_text']}\" (API Identified: {evidence['gt_lang']})")

    decision = vote_whisper_samples(samples)
    print(f"     [Final Verdict] Whisper Decision: {decision['language']} | Reason: {decision['reason']}")
    
    return {
        "Lang_Predicted": decision["language"],
        "Lang_Is_Match": str(True if not expected_language else decision["language"] == expected_language),
        "Lang_Method": "WHISPER_VOTE" if decision["accepted"] else "HUMAN_REVIEW",
        "Lang_Verdict": "WHISPER_LANGUAGE_FOUND" if decision["accepted"] else "HUMAN_REVIEW",
        "Lang_Reason": decision["reason"],
        "GT_Title_Lang": gt_title_lang,
        "GT_Audio_Lang": " | ".join(s["gt_lang"] for s in samples if s["gt_lang"])
    }

def empty_language_guess():
    return {
        "Lang_Predicted": "", 
        "Lang_Is_Match": "", 
        "Lang_Method": "", 
        "Lang_Verdict": "", 
        "Lang_Reason": "",
        "GT_Title_Lang": "", 
        "GT_Audio_Lang": ""
    }


# ==========================================
# 6. PIPELINE CONTROLLER
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
    output_template = str(download_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        # AUDIO ONLY: Safe to download full length without hitting the 90MB cap
        "format": "worstaudio[ext=m4a]/worst",
        "max_filesize": MAX_DOWNLOAD_BYTES,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    print(f"  -> Downloading full audio track...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        duration = info.get("duration", 0) 

    path = Path(filename)
    if not path.exists():
        matches = list(download_dir.glob(f"{info.get('id', '*')}.*"))
        if matches: path = matches[0]
    
    return path, duration

def run_isolated_language_test(
    data_path=PROJECT_ROOT / "data" / "output_claims.csv",
    output_path=PROJECT_ROOT / "data" / "language_evaluation_report.csv",
):
    print("=====================================================")
    print("     STARTING STANDALONE LANGUAGE EVALUATION ENGINE  ")
    print("=====================================================")
    
    load_env_file()
    if not os.environ.get("GOOGLE_API_KEY"):
        print("[!] Warning: GOOGLE_API_KEY environment variable is blank. Cloud translation modules will fail.")

    df = pd.read_csv(data_path)
    if AUTO_YES_COLUMN not in df.columns or YOUTUBE_COLUMN not in df.columns:
        raise KeyError(f"Input file missing crucial tracking keys: '{AUTO_YES_COLUMN}' or '{YOUTUBE_COLUMN}'")

    # Focus exclusively on target action rows
    eval_df = df[df[AUTO_YES_COLUMN].eq("Auto Yes")].copy()
    
    total_candidates = len(eval_df)
    print(f"[*] Found {total_candidates} validation candidates to process.")

    # Apply structure columns dynamically
    for col in empty_language_guess().keys(): 
        eval_df[col] = ""

    whisper_model = load_whisper_model()
    success_count = 0

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        
        for row_number, row in eval_df.iterrows():
            url = normalize_youtube_url(row[YOUTUBE_COLUMN])
            title = str(row.get(VIDEO_TITLE_COLUMN, ""))
            
            # 1. Check for expected language and intercept Pandas "nan" explicitly
            expected_lang = str(row.get(EXPECTED_LANGUAGE_COLUMN, "")).strip()
            if expected_lang.lower() == 'nan':
                expected_lang = ""

            # 2. If Expected Language is missing, translate the WESS ID using our dynamic dictionary
            if not expected_lang and WESS_LANGUAGE_ID_COLUMN in row:
                raw_id = str(row[WESS_LANGUAGE_ID_COLUMN]).split('.')[0].strip()
                if raw_id.lower() != 'nan':
                    expected_lang = WESS_ID_TO_LANGUAGE.get(raw_id, raw_id)
            
            # 3. Write it back to the dataframe IMMEDIATELY so visualize.py can see it
            eval_df.at[row_number, EXPECTED_LANGUAGE_COLUMN] = expected_lang

            print(f"\n=====================================================")
            print(f"Processing candidate validation link: {url}")
            
            if not url:
                result = empty_language_guess()
                result["Lang_Reason"] = "Blank or corrupt URL index configuration."
            else:
                try:
                    video_path, duration = download_video(url, temp_dir)
                    
                    result = run_language_checker(
                        video_path=video_path,
                        duration_seconds=duration,
                        expected_language=expected_lang,
                        title=title,
                        whisper_model=whisper_model,
                        workdir=temp_dir
                    )
                    
                    success_count += 1
                    print(f"[*] [{success_count}/{total_candidates}] Evaluation pipeline execution complete.")
                    
                except Exception as exc:
                    print(f"  -> [CRITICAL FAILURE] Bypassing target asset {url} due to error: {exc}")
                    result = empty_language_guess()
                    result["Lang_Reason"] = f"Fatal Execution Error: {exc}"

            # Save the result details into our active row
            for column, value in result.items():
                eval_df.at[row_number, column] = value

            # Real-time Checkpoint Saving: Write changes immediately after every iteration
            eval_df.to_csv(output_path, index=False)

    print(f"\n=====================================================")
    print(f"[*] EVALUATION SEQUENCE COMPLETE!")
    print(f"[*] Clean data compiled and saved to: {output_path}")
    print("=====================================================")

if __name__ == "__main__":
    run_isolated_language_test()