#!/usr/bin/env python3
"""
============================================================
  FOOTAGE TOOL — RAILWAY SERVER v2
  Improved: batch upload, parallel processing, better accuracy
  via negative prompts and multi-variant scene descriptions
============================================================
"""

import os
import json
import uuid
import base64
import threading
import ssl
import re
import concurrent.futures
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="web")
jobs = {}

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

def expand_description(desc):
    variants = [
        desc,
        f"a video of {desc.lower()}",
        f"a photo of {desc.lower()}",
        f"footage showing {desc.lower()}",
        f"television commercial showing {desc.lower()}"
    ]
    return variants[:5]

NEGATIVE_PROMPTS = [
    "blurry unfocused footage",
    "dark underexposed video",
    "overexposed washed out video",
    "empty room with no people",
    "black screen or blank frame",
    "abstract background texture",
]

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/api/start", methods=["POST"])
def start_job():
    data        = request.json
    job_id      = str(uuid.uuid4())[:8]
    project     = data.get("project_name", "Project").strip().replace(" ", "_")
    script_text = data.get("script_text", "").strip()
    threshold   = float(data.get("threshold", 0.20))

    if not script_text:
        return jsonify({"error": "Scene descriptions are empty"}), 400

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
        "status":    "waiting",
        "project":   project,
        "scenes":    scenes,
        "threshold": threshold,
        "clips":     {},
        "results":   None,
        "log":       [f"Job created: {project}", f"{len(scenes)} scenes parsed", "Waiting for frames..."]
    }
    return jsonify({"job_id": job_id, "scenes": scenes})

@app.route("/api/upload_batch", methods=["POST"])
def upload_batch():
    data   = request.json
    job_id = data.get("job_id")
    clips  = data.get("clips", [])
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    for clip in clips:
        jobs[job_id]["clips"][clip["clip_name"]] = {
            "frames":   clip.get("frames", []),
            "metadata": clip.get("metadata", {})
        }
    jobs[job_id]["status"] = "receiving"
    jobs[job_id]["log"].append(f"Received batch: {len(clips)} clips ({len(jobs[job_id]['clips'])} total)")
    return jsonify({"ok": True})

@app.route("/api/upload_frames", methods=["POST"])
def upload_frames():
    data      = request.json
    job_id    = data.get("job_id")
    clip_name = data.get("clip_name")
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    jobs[job_id]["clips"][clip_name] = {
        "frames":   data.get("frames", []),
        "metadata": data.get("metadata", {})
    }
    jobs[job_id]["status"] = "receiving"
    return jsonify({"ok": True})

def analyze_clip(clip_name, clip_data, scenes, threshold):
    frames_b64        = clip_data["frames"]
    meta              = clip_data["metadata"]
    scene_descriptions = [s["description"] for s in scenes]
    scene_labels       = [s["scene"]       for s in scenes]

    pil_images    = []
    blur_scores   = []
    bright_scores = []

    for fb64 in frames_b64:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(fb64))).convert("RGB")
            pil_images.append(img)
            arr  = np.array(img)
            gray = np.mean(arr, axis=2)
            blur_scores.append(float(np.var(np.diff(np.diff(gray, axis=0), axis=1))))
            bright_scores.append(float(arr.mean()))
        except:
            pass

    if not pil_images:
        return {"clip": clip_name, "usable": False, "label_color": "Red", "error": "No frames", **meta}

    all_texts   = []
    scene_count = []
    for desc in scene_descriptions:
        variants = expand_description(desc)
        all_texts.extend(variants)
        scene_count.append(len(variants))
    neg_start = len(all_texts)
    all_texts.extend(NEGATIVE_PROMPTS)

    try:
        inputs = processor(text=all_texts, images=pil_images, return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            avg_probs = model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy().mean(axis=0)

        scene_scores = []
        idx = 0
        for count in scene_count:
            scene_scores.append(float(avg_probs[idx:idx+count].mean()))
            idx += count

        neg_score = float(avg_probs[neg_start:].mean())
        adjusted  = [s * (1 - neg_score * 2) for s in scene_scores]

        sorted_adj = sorted(adjusted, reverse=True)
        best_idx   = int(np.argmax(adjusted))
        best_score = adjusted[best_idx]
        best_scene = scene_labels[best_idx]
        best_desc  = scene_descriptions[best_idx]

        # Penalize uncertain matches
        if len(sorted_adj) > 1 and (sorted_adj[0] - sorted_adj[1]) < 0.05:
            best_score *= 0.7

    except Exception as e:
        return {"clip": clip_name, "usable": False, "label_color": "Red", "error": str(e), **meta}

    avg_blur   = sum(blur_scores)   / len(blur_scores)   if blur_scores   else 0
    avg_bright = sum(bright_scores) / len(bright_scores) if bright_scores else 128
    issues     = []
    if avg_blur < 50:    issues.append("blurry")
    if avg_bright < 30:  issues.append("too dark")
    if avg_bright > 225: issues.append("overexposed")
    quality    = "good" if not issues else ", ".join(issues)
    usable     = quality == "good" and best_score >= threshold

    label_color = "Green" if (usable and best_score >= threshold * 2) else "Yellow" if usable else "Red"

    scene_breakdown = sorted([
        {"scene": scene_labels[i], "description": scene_descriptions[i], "score": round(float(scene_scores[i]), 4)}
        for i in range(len(scenes))
    ], key=lambda x: x["score"], reverse=True)

    return {
        "clip": clip_name, "path": meta.get("path",""),
        "usable": usable, "label_color": label_color,
        "best_scene": best_scene, "best_scene_desc": best_desc,
        "match_score": round(best_score, 4),
        "avg_blur": round(avg_blur, 2), "avg_brightness": round(avg_bright, 2),
        "quality": quality, "scene_breakdown": scene_breakdown[:3],
        "duration_sec": meta.get("duration_sec", 0),
        "fps": meta.get("fps", 25), "total_frames": meta.get("total_frames", 0)
    }

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
        job["log"].append(f"Analyzing {len(clips)} clips with improved matching...")
        all_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(analyze_clip, n, d, scenes, threshold): n
                      for n, d in clips.items()}
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                all_results.append(r)
                job["log"].append(
                    f"[{r.get('label_color','Red')}] {r['clip']} → "
                    f"{r.get('best_scene','?')} (score: {r.get('match_score',0):.2f}, {r.get('quality','?')})"
                )

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
            "script_file": job["project"], "footage_folder": "",
            "threshold": threshold, "total_clips": len(all_results),
            "usable_clips": sum(1 for r in all_results if r.get("usable")),
            "rejected_clips": sum(1 for r in all_results if not r.get("usable")),
            "scenes": scenes, "scene_bins": scene_bins,
            "rejected": rejected, "all_clips": all_results
        }
        job["results"] = report
        job["status"]  = "done"
        job["log"].append(f"✅ Done! {report['usable_clips']} usable / {report['rejected_clips']} rejected")

    threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/job/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    return jsonify({"status": job["status"], "log": job["log"],
                    "results": job["results"], "project": job["project"],
                    "scenes": job["scenes"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
