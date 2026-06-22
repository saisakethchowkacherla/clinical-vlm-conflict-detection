"""Test the conflict detection path: ABSENT image + high lab → PRESENT multimodal → DEFER."""
import json
import urllib.request
from pathlib import Path

# Normal-100.png hashes to ABSENT image-only; high lab flips multimodal → conflict
IMG = Path("data/covid19-radiography-database/COVID-19_Radiography_Dataset/Normal/images/Normal-100.png")
img_bytes = IMG.read_bytes()

sep = b"------B\r\n"
body = (
    sep + b'Content-Disposition: form-data; name="lab_value"\r\n\r\n14.0\r\n'
    + sep + b'Content-Disposition: form-data; name="ref_low"\r\n\r\n4.0\r\n'
    + sep + b'Content-Disposition: form-data; name="ref_high"\r\n\r\n11.0\r\n'
    + sep
    + b'Content-Disposition: form-data; name="image"; filename="Normal-100.png"\r\n'
    + b"Content-Type: image/png\r\n\r\n"
    + img_bytes
    + b"\r\n------B--\r\n"
)

req = urllib.request.Request(
    "http://127.0.0.1:8080/predict",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=----B"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print(json.dumps(json.loads(resp.read()), indent=2))
