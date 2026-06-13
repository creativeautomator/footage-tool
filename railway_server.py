#!/usr/bin/env python3
"""
============================================================
  FOOTAGE TOOL — RAILWAY SERVER v2.2
  Added: specialized prompts for product shots and interviews
  Tags: [PRODUCT], [PRODUCT-USE], [PRODUCT-BRAND], [INTERVIEW]
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

# ── Shot type variant library ─────────────────────────────────
# Each tag maps to a set of CLIP-friendly descriptions
# Combined with the user's own description for best results

SHOT_TYPE_VARIANTS = {
    "[PRODUCT]": [
        "a single product bottle on a clean white background",
        "product packaging on white studio background",
        "commercial product shot with clean background",
        "isolated product photography studio shot",
        "product on white background with soft lighting",
        "close-up of product label and packaging",
        "commercial advertisement product display",
    ],
    "[PRODUCT-USE]": [
        "a person using a product in their hands",
        "someone holding and applying a product",
        "person demonstrating how to use a product",
        "hands holding a product close up",
        "person washing or applying product to skin",
        "lifestyle shot of someone using a product",
        "person interacting with consumer product",
    ],
    "[PRODUCT-BRAND]": [
        "product logo and brand name clearly visible",
        "brand logo on packaging close up",
        "commercial with brand name prominently displayed",
        "product label with brand identity",
        "close-up of product with visible branding",
        "advertising shot showing brand and product together",
        "product with logo centered in frame",
    ],
    "[INTERVIEW]": [
        "a person talking directly to the camera",
        "talking head interview with person facing camera",
        "sit-down interview with person speaking",
        "person giving interview on camera",
        "close-up of person's face talking to camera",
        "interview shot with shallow depth of field",
        "person speaking in a documentary style interview",
        "medium shot of person talking to interviewer",
    ],
}

def get_variants(description):
    """
    Parse scene description for shot type tags and return
    expanded CLIP prompt variants.
    Tags: [PRODUCT], [PRODUCT-USE], [PRODUCT-BRAND], [INTERVIEW]
    """
    desc_clean = description
    detected_tags = []

    for tag in SHOT_TYPE_VARIANTS:
        if tag.lower() in description.lower():
            detected_tags.append(tag)
            desc_clean = re.sub(re.escape(tag), "", desc_clean, flags=re.IGNORECASE).strip()

    variants = []

    if detected_tags:
        # Add base description (cleaned)
        if desc_clean:
            variants.append(desc_clean)
            variants.append(f"a video of {desc_clean.lower()}")
            variants.append(f"footage showing {desc_clean.lower()}")

        # Add shot-type specific variants
        for tag in detected_tags:
            tag_variants = SHOT_TYPE_VARIANTS[tag]
            # Combine tag variants with the specific description
            for tv in tag_variants[:4]:  # top 4 per tag
                if desc_clean:
                    variants.append(f"{tv} — {desc_clean.lower()}")
                else:
                    variants.append(tv)

        # Add remaining tag variants without description
        for tag in detected_tags:
            variants.extend(SHOT_TYPE_VARIANTS[tag][4:])

    else:
        # No tag — use standard 3 variants
        variants = [
            description,
            f"a video of {description.lower()}",
            f"footage showing {description.lower()}",
        ]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique[:10]  # max 10 variants

def detect_shot_type(description):
    """Return human-readable shot type label for logging."""
    for tag in SHOT_TYPE_VARIANTS:
        if tag.lower() in description.lower():
            return tag.replace("[","").replace("]","")
    return "GENERAL"

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

    # Log detected shot types
    shot_types = [detect_shot_type(s["description"]) for s in scenes]

    jobs[job_id] = {
        "status":    "waiting",
        "project":   project,
        "scenes":    scenes,
        "threshold": threshold,
        "clips":     {},
        "results":   None,
        "log":       [
            f"Job created: {project}",
            f"{len(scenes)} scenes parsed",
            f"Shot types detected: {', '.join(set(shot_types))}",
            "Waiting for frames..."
        ]
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
    jobs[job_id]["log"].append(
        f"Received batch: {len(clips)} clips "
        f"({len(jobs[job_id]['clips'])} total)"
    )
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
    frames_b64         = clip_data["frames"]
    meta               = clip_data["metadata"]
    scene_descriptions = [s["description"] for s in scenes]
    scene_labels       = [s["scene"]       for s in scenes]

    # Decode frames
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
        return {"clip": clip_name, "usable": False, "label_color": "Red",
                "error": "No frames decoded", **meta}

    # Build text prompts — expanded variants per scene
    all_texts   = []
    scene_count = []
    for desc in scene_descriptions:
        variants = get_variants(desc)
        all_texts.extend(variants)
        scene_count.append(len(variants))

    try:
        inputs = processor(
            text=all_texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        with torch.no_grad():
            avg_probs = model(**inputs).logits_per_image.softmax(
                dim=-1).cpu().numpy().mean(axis=0)

        # Average scores per scene across all variants
        scene_scores = []
        idx = 0
        for count in scene_count:
            scene_scores.append(float(avg_probs[idx:idx+count].mean()))
            idx += count

        best_idx   = int(np.argmax(scene_scores))
        best_score = scene_scores[best_idx]
        best_scene = scene_labels[best_idx]
        best_desc  = scene_descriptions[best_idx]

    except Exception as e:
        return {"clip": clip_name, "usable": False, "label_color": "Red",
                "error": str(e), **meta}

    # Quality checks
    avg_blur   = sum(blur_scores)   / len(blur_scores)   if blur_scores   else 0
    avg_bright = sum(bright_scores) / len(bright_scores) if bright_scores else 128
    issues = []
    if avg_blur < 50:    issues.append("blurry")
    if avg_bright < 30:  issues.append("too dark")
    if avg_bright > 225: issues.append("overexposed")
    quality = "good" if not issues else ", ".join(issues)

    usable = (quality == "good") and (best_score >= threshold)

    if usable and best_score >= threshold * 2:
        label_color = "Green"
    elif usable:
        label_color = "Yellow"
    else:
        label_color = "Red"

    scene_breakdown = sorted([
        {"scene": scene_labels[i], "description": scene_descriptions[i],
         "score": round(float(scene_scores[i]), 4)}
        for i in range(len(scenes))
    ], key=lambda x: x["score"], reverse=True)

    return {
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
        job["log"].append(f"Analyzing {len(clips)} clips...")

        all_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(analyze_clip, name, data, scenes, threshold): name
                for name, data in clips.items()
            }
            for future in concurrent.futures.as_completed(futures):
                r = future.result()
                all_results.append(r)
                job["log"].append(
                    f"[{r.get('label_color','Red')}] {r['clip']} → "
                    f"{r.get('best_scene','?')} "
                    f"(score: {r.get('match_score',0):.2f}, "
                    f"{r.get('quality','?')})"
                )

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
        job["log"].append(
            f"✅ Done! {report['usable_clips']} usable / "
            f"{report['rejected_clips']} rejected"
        )

    threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify({"ok": True})

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
