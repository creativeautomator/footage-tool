#!/usr/bin/env python3
"""
============================================================
  FOOTAGE TOOL — LOCAL AGENT
  Runs on each teammate's Mac.
  Extracts frames from local footage and sends them to
  the Railway server for analysis.
  Footage never fully uploads — only small JPEG snapshots.

  HOW TO RUN:
    python3 agent.py \
      --server   https://your-app.railway.app \
      --job      JOB_ID \
      --footage  /path/to/raw/footage \
      --output   /path/to/save/results

  The job ID comes from the web interface after you
  paste your scene descriptions and click Start.
============================================================
"""

import os
import sys
import json
import base64
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import cv2
    import numpy as np
    from PIL import Image
    import urllib.request
    import urllib.parse
    import io
except ImportError as e:
    print(f"\n❌  Missing package: {e}")
    print("Run: pip3 install opencv-python pillow numpy")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--server",  required=True, help="Railway server URL e.g. https://your-app.railway.app")
parser.add_argument("--job",     required=True, help="Job ID from the web interface")
parser.add_argument("--footage", required=True, help="Path to raw footage folder")
parser.add_argument("--output",  required=True, help="Path to save Organized Footage")
parser.add_argument("--frames",  type=int, default=8, help="Frames to sample per clip")
parser.add_argument("--ext",     nargs="+", default=["mp4","mov","avi","mxf","r3d"])
args = parser.parse_args()

SERVER   = args.server.rstrip("/")
JOB_ID   = args.job
FOOTAGE  = os.path.expanduser(args.footage)
OUTPUT   = os.path.expanduser(args.output)
N_FRAMES = args.frames

# ── Helper: POST JSON ────────────────────────────────────────
def post_json(url, data):
    body    = json.dumps(data).encode("utf-8")
    req     = urllib.request.Request(url, data=body,
              headers={"Content-Type": "application/json"}, method="POST")
    ctx     = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        return json.loads(r.read().decode())

def get_json(url):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx, timeout=30) as r:
        return json.loads(r.read().decode())

# ── Collect clips ────────────────────────────────────────────
def collect_clips(folder, extensions):
    clips = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if any(f.lower().endswith("." + e) for e in extensions):
                clips.append(os.path.join(root, f))
    return clips

# ── Extract frames ───────────────────────────────────────────
def extract_frames(clip_path, n=8):
    cap   = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        cap.release()
        return [], {}

    indices = [int(total * i / n) for i in range(n)]
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(rgb)
            img.thumbnail((224, 224))  # small size — fast upload
            buf  = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode())

    cap.release()
    duration = total / fps
    return frames_b64, {
        "path":         clip_path,
        "duration_sec": round(duration, 2),
        "fps":          round(fps, 2),
        "total_frames": total
    }

# ── Main ─────────────────────────────────────────────────────
print(f"\n🎬  FOOTAGE TOOL — LOCAL AGENT")
print(f"{'='*50}")
print(f"  Server:  {SERVER}")
print(f"  Job ID:  {JOB_ID}")
print(f"  Footage: {FOOTAGE}")
print(f"{'='*50}\n")

# Collect clips
clips = collect_clips(FOOTAGE, args.ext)
if not clips:
    print(f"❌  No video files found in: {FOOTAGE}")
    sys.exit(1)

print(f"  Found {len(clips)} clip(s)\n")
print(f"  Extracting frames and sending to server...")
print(f"  (Only small JPEG snapshots are sent — footage stays on your Mac)\n")

# Send frames for each clip
for i, clip_path in enumerate(clips):
    clip_name = os.path.basename(clip_path)
    print(f"  [{i+1}/{len(clips)}] {clip_name}...", end=" ", flush=True)

    frames_b64, meta = extract_frames(clip_path, N_FRAMES)
    if not frames_b64:
        print("⚠️  skipped (could not read)")
        continue

    try:
        post_json(f"{SERVER}/api/upload_frames", {
            "job_id":    JOB_ID,
            "clip_name": clip_name,
            "frames":    frames_b64,
            "metadata":  meta
        })
        print(f"✅  ({len(frames_b64)} frames sent)")
    except Exception as e:
        print(f"❌  Error: {e}")

# Signal analysis to start
print(f"\n  All frames sent. Starting analysis on server...")
try:
    post_json(f"{SERVER}/api/analyze/{JOB_ID}", {})
except Exception as e:
    print(f"❌  Could not start analysis: {e}")
    sys.exit(1)

# Poll for results
import time
print(f"  Waiting for results", end="", flush=True)
last_log_len = 0

while True:
    time.sleep(3)
    try:
        data = get_json(f"{SERVER}/api/job/{JOB_ID}")
        # Print new log lines
        new_lines = data["log"][last_log_len:]
        for line in new_lines:
            print(f"\n  {line}", end="", flush=True)
        last_log_len = len(data["log"])

        if data["status"] == "done":
            print(f"\n")
            report = data["results"]
            break
        elif data["status"] == "error":
            print(f"\n❌  Analysis failed on server.")
            sys.exit(1)
        else:
            print(".", end="", flush=True)
    except Exception as e:
        print(f"\n  ⚠️  Poll error: {e}")
        time.sleep(5)

# ── Save results and organize footage ───────────────────────
import shutil

os.makedirs(OUTPUT, exist_ok=True)

# Save report.json
report_path = os.path.join(OUTPUT, "report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

# Create organized folders
organized = os.path.join(OUTPUT, "Organized Footage")
os.makedirs(organized, exist_ok=True)

scenes     = report.get("scenes", [])
scene_bins = report.get("scene_bins", {})
rejected   = report.get("rejected", [])

for scene in scenes:
    sname   = scene["scene"]
    sdesc   = scene["description"][:40].replace("/","-").replace(":","-")
    folder  = os.path.join(organized, f"{sname} - {sdesc}")
    os.makedirs(folder, exist_ok=True)
    for clip in scene_bins.get(sname, []):
        src = clip.get("path","")
        dst = os.path.join(folder, clip["clip"])
        if src and os.path.exists(src) and not os.path.exists(dst):
            try:    os.symlink(src, dst)
            except: shutil.copy2(src, dst)

rej_folder = os.path.join(organized, "REJECTED")
os.makedirs(rej_folder, exist_ok=True)
for clip in rejected:
    src = clip.get("path","")
    dst = os.path.join(rej_folder, clip["clip"])
    if src and os.path.exists(src) and not os.path.exists(dst):
        try:    os.symlink(src, dst)
        except: pass

# ── Summary ──────────────────────────────────────────────────
print(f"{'='*50}")
print(f"  ✅  DONE!")
print(f"{'='*50}")
print(f"  Total clips:   {report['total_clips']}")
print(f"  Usable clips:  {report['usable_clips']}")
print(f"  Rejected:      {report['rejected_clips']}")
print(f"\n  Organized Footage saved to:")
print(f"  {organized}")
print(f"\n  To import into Premiere:")
print(f"  File → Import → select 'Organized Footage' folder")
print(f"{'='*50}\n")
