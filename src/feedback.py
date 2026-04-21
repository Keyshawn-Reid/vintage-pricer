import csv
import hashlib
import os
from datetime import datetime, timezone

FEEDBACK_DIR = "data/feedback"


def compute_image_ref(image_path):
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:16]


def save_feedback(brand, image_ref, ai_values, user_values):
    """
    Append one row per form submission to data/feedback/<brand>_feedback.csv.
    Only called when a photo was analyzed (image_ref is non-empty).
    Stores AI prediction, user value, and whether the user corrected it for each field.
    """
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    path = os.path.join(FEEDBACK_DIR, f"{brand}_feedback.csv")

    all_fields = sorted(set(list(ai_values.keys()) + list(user_values.keys())))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "brand": brand,
        "image_ref": image_ref,
    }
    for field in all_fields:
        ai_val   = str(ai_values.get(field, "")).strip()
        user_val = str(user_values.get(field, "")).strip()
        row[f"ai_{field}"]        = ai_val
        row[f"user_{field}"]      = user_val
        row[f"corrected_{field}"] = int(ai_val != user_val and ai_val != "")

    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
