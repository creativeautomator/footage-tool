#!/usr/bin/env python3
"""
============================================================
  FOOTAGE TOOL — RAILWAY SERVER
  Hosted backend for the Footage Analysis Tool.
  Receives frames from local agents, runs CLIP matching,
  returns results.

  DEPLOY TO RAILWAY:
  1. Create account at railway.app
  2. New Project → Deploy from GitHub
  3. Upload this folder
  4. Railway auto-detects Python and deploys

  ENVIRONMENT VARIABLES (set in Railway dashboard):
  None required — works out of the box.
============================================================
"""

import os
import json
import uuid
import base64
import threading
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="web")

# In-memory job store
jobs = {}

# ── Load CLIP model once at startup ──────────────────────────
print("Loading CLIP model...")
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import io

device = "cuda" if torch.cuda.is_available() else "cpu"
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"CLIP loaded on {device}")

# ── Route: serve web UI ──────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("web", "index.html")

# ── Route: start a job ───────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def start_job():
    data         = request.json
    job_id       = str(uuid.uuid4())[:8]
    project      = data.get("project_name", "Project").strip().replace(" ", "_")
    script_text  = data.get("script_text", "").strip()
    threshold    = float(data.get("threshold", 0.20))

    if not script_text:
        return jsonify({"error": "Scene descriptions are empty"}), 400

    # Parse scenes
    import re
    lines  = [l.strip() for l in script_text.split("\n") if l.strip()]
    scenes = []
    for line in lines:
        m = re.match(r'^(?:scene|shot|sc)\s*[\.\-]?\s*(\d+)\s*[\:\-]\s*(.+)', line, re.IGNORECASE)
        n = re.match(r'^(\d+)\.\s+(.+)', line)
        if m:
            scenes.append({"scene": f"Scene {m.group(1)}", "description": m.group(2).strip()})
        elif n:
            scenes.append({"scene": f"Scene {n.group(1)}", "description": n.group(2).strip()})
        else:
            scenes.append({"scene": f"Scene {len(scenes)+1}", "description": line})

    jobs[job_id] = {
        "status":     "waiting",
        "project":    project,
        "scenes":     scenes,
        "threshold":  threshold,
        "clips":      {},      # clip_name → {frames, metadata}
        "results":    None,
        "log":        [f"Job created for: {project}", f"{len(scenes)} scenes parsed", "Waiting for frames from your Mac..."]
    }

    return jsonify({"job_id": job_id, "scenes": scenes})

# ── Route: agent uploads frames for a clip ──────────────────
@app.route("/api/upload_frames", methods=["POST"])
def upload_frames():
    data      = request.json
    job_id    = data.get("job_id")
    clip_name = data.get("clip_name")
    frames_b64 = data.get("frames", [])   # list of base64 JPEG strings
    metadata  = data.get("metadata", {})  # duration, fps, total_frames

    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    jobs[job_id]["clips"][clip_name] = {
        "frames":   frames_b64,
        "metadata": metadata
    }
    jobs[job_id]["log"].append(f"Received frames: {clip_name}")
    jobs[job_id]["status"] = "receiving"

    return jsonify({"ok": True})

# ── Route: agent signals all frames sent, start analysis ─────
@app.route("/api/analyze/<job_id>", methods=["POST"])
def analyze(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    def run_analysis():
        job       = jobs[job_id]
        scenes    = job["scenes"]
        threshold = job["threshold"]
        clips     = job["clips"]

        job["status"] = "analyzing"
        job["log"].append(f"Starting analysis of {len(clips)} clips...")

        scene_descriptions = [s["description"] for s in scenes]
        scene_labels       = [s["scene"]       for s in scenes]

        all_results = []

        for clip_name, clip_data in clips.items():
            frames_b64 = clip_data["frames"]
            meta       = clip_data["metadata"]

            # Decode frames
            pil_images = []
            blur_scores   = []
            bright_scores = []

            for fb64 in frames_b64:
                try:
                    img_bytes = base64.b64decode(fb64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_images.append(img)

                    # Quality checks on numpy array
                    arr = np.array(img)
                    gray = np.mean(arr, axis=2)
                    # Blur via Laplacian variance approximation
                    blur = float(np.var(np.diff(np.diff(gray, axis=0), axis=1)))
                    blur_scores.append(blur)
                    bright_scores.append(float(arr.mean()))
                except:
                    pass

            if not pil_images:
                all_results.append({
                    "clip": clip_name, "usable": False,
                    "label_color": "Red", "error": "No frames decoded",
                    **meta
                })
                continue

            # CLIP matching
            try:
                inputs = processor(
                    text=scene_descriptions,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()

                avg_scores = probs.mean(axis=0)
                best_idx   = int(avg_scores.argmax())
                best_score = float(avg_scores[best_idx])
                best_scene = scene_labels[best_idx]
                best_desc  = scene_descriptions[best_idx]

            except Exception as e:
                all_results.append({
                    "clip": clip_name, "usable": False,
                    "label_color": "Red", "error": str(e), **meta
                })
                continue

            # Quality
            avg_blur   = sum(blur_scores)   / len(blur_scores)
            avg_bright = sum(bright_scores) / len(bright_scores)
            issues = []
            if avg_blur < 50:        issues.append("blurry")
            if avg_bright < 30:      issues.append("too dark")
            if avg_bright > 225:     issues.append("overexposed")
            quality = "good" if not issues else ", ".join(issues)

            quality_ok = quality == "good"
            match_ok   = best_score >= threshold
            usable     = quality_ok and match_ok

            if usable and best_score >= threshold * 2:
                label_color = "Green"
            elif usable:
                label_color = "Yellow"
            else:
                label_color = "Red"

            scene_breakdown = [
                {"scene": scene_labels[i], "description": scene_descriptions[i], "score": round(float(avg_scores[i]), 4)}
                for i in range(len(scenes))
            ]
            scene_breakdown.sort(key=lambda x: x["score"], reverse=True)

            all_results.append({
                "clip":            clip_name,
                "path":            meta.get("path", ""),
                "usable":          usable,
                "label_color":     label_color,
                "best_scene":      best_scene,
                "best_scene_desc": best_desc,
                "match_score":     round(best_score, 4),
                "avg_blur":        round(avg_blur, 2),
                "avg_brightness":  round(avg_bright, 2),
                "quality":         quality,
                "scene_breakdown": scene_breakdown[:3],
                "duration_sec":    meta.get("duration_sec", 0),
                "fps":             meta.get("fps", 25),
                "total_frames":    meta.get("total_frames", 0)
            })

            job["log"].append(f"[{label_color}] {clip_name} → {best_scene} (score: {best_score:.2f}, {quality})")

        # Build report
        scene_bins = {s["scene"]: [] for s in scenes}
        rejected   = []
        for r in all_results:
            if r.get("usable"):
                scene_bins[r["best_scene"]].append(r)
            else:
                rejected.append(r)
        for s in scene_bins:
            scene_bins[s].sort(key=lambda x: x["match_score"], reverse=True)

        report = {
            "script_file":    job["project"],
            "footage_folder": "",
            "threshold":      threshold,
            "total_clips":    len(all_results),
            "usable_clips":   sum(1 for r in all_results if r.get("usable")),
            "rejected_clips": sum(1 for r in all_results if not r.get("usable")),
            "scenes":         scenes,
            "scene_bins":     scene_bins,
            "rejected":       rejected,
            "all_clips":      all_results
        }

        job["results"] = report
        job["status"]  = "done"
        job["log"].append(f"✅ Done! {report['usable_clips']} usable clips found.")

    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()

    return jsonify({"ok": True})

# ── Route: poll job ──────────────────────────────────────────
@app.route("/api/job/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    return jsonify({
        "status":  job["status"],
        "log":     job["log"],
        "results": job["results"],
        "project": job["project"],
        "scenes":  job["scenes"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
