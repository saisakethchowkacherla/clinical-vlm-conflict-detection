"""Smoke-test every validation check against the live server."""
import io
import json
import urllib.request
import urllib.error

from PIL import Image

BASE = "http://127.0.0.1:8080"
VALID_IMG = "data/covid19-radiography-database/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png"
BOUNDARY = "----ValBoundary"
SEP = f"--{BOUNDARY}\r\n".encode()
END = f"--{BOUNDARY}--\r\n".encode()


def build_body(img_bytes: bytes, filename: str, content_type: str = "image/png") -> bytes:
    return (
        SEP
        + b'Content-Disposition: form-data; name="lab_value"\r\n\r\n7.0\r\n'
        + SEP
        + b'Content-Disposition: form-data; name="ref_low"\r\n\r\n4.5\r\n'
        + SEP
        + b'Content-Disposition: form-data; name="ref_high"\r\n\r\n11.0\r\n'
        + SEP
        + f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
        + f"Content-Type: {content_type}\r\n\r\n".encode()
        + img_bytes
        + b"\r\n"
        + END
    )


def post(img_bytes: bytes, filename: str, content_type: str = "image/png") -> tuple[int, dict]:
    body = build_body(img_bytes, filename, content_type)
    req = urllib.request.Request(
        f"{BASE}/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def make_png(w: int, h: int, rgb: tuple = (128, 128, 128)) -> bytes:
    img = Image.new("RGB", (w, h), rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


valid_bytes = open(VALID_IMG, "rb").read()

cases = [
    ("Valid X-ray",           lambda: post(valid_bytes, "xray.png"),                        200, None),
    ("Wrong extension (.gif)",lambda: post(valid_bytes, "xray.gif", "image/gif"),            400, None),
    ("Too small (50x50)",     lambda: post(make_png(50, 50), "small.png"),                  400, None),
    ("Too large (6000x6000)", lambda: post(make_png(6000, 6000), "huge.png"),               400, None),
    ("Too wide (900x100)",    lambda: post(make_png(900, 100), "wide.png"),                 400, None),
    ("Colourful (red image)", lambda: post(make_png(300, 300, (220, 20, 20)), "red.png"),   400, "does not appear"),
]

print(f"{'Case':<35} {'Status':>6}  {'OK?':<5}  Message")
print("-" * 90)
for name, fn, expected_status, expected_msg in cases:
    status, body = fn()
    ok = status == expected_status
    if expected_msg:
        ok = ok and expected_msg in json.dumps(body)
    msg = body.get("detail", {}).get("message", body.get("message", body.get("validation_passed", "")))
    print(f"{name:<35} {status:>6}  {'PASS' if ok else 'FAIL':<5}  {str(msg)[:60]}")
