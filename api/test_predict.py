"""Quick smoke test for the /predict endpoint."""
import json
import urllib.request
from pathlib import Path

IMG = Path("data/covid19-radiography-database/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png")
img_bytes = IMG.read_bytes()

boundary = b"----TestBoundary1234"
sep = b"--" + boundary + b"\r\n"
end = b"--" + boundary + b"--\r\n"

body = (
    sep
    + b'Content-Disposition: form-data; name="lab_value"\r\n\r\n12.5\r\n'
    + sep
    + b'Content-Disposition: form-data; name="ref_low"\r\n\r\n4.0\r\n'
    + sep
    + b'Content-Disposition: form-data; name="ref_high"\r\n\r\n11.0\r\n'
    + sep
    + b'Content-Disposition: form-data; name="image"; filename="COVID-1.png"\r\n'
    + b"Content-Type: image/png\r\n\r\n"
    + img_bytes
    + b"\r\n"
    + end
)

req = urllib.request.Request(
    "http://127.0.0.1:8080/predict",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=----TestBoundary1234"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print(json.dumps(json.loads(resp.read()), indent=2))
