"""Check saturation + grayscale ratio for X-rays and various non-X-ray images."""
import io
from PIL import Image


def stats(source):
    img = Image.open(source) if isinstance(source, str) else Image.open(io.BytesIO(source))
    img = img.convert("RGB")
    pixels = list(img.getdata())
    total = len(pixels)

    hsv = img.convert("HSV")
    _, s, _ = hsv.split()
    avg_sat = sum(s.getdata()) / total / 255.0
    near_gray = sum(1 for r, g, b in pixels if max(r, g, b) - min(r, g, b) < 30)
    gray_ratio = near_gray / total
    sat_ok = avg_sat <= 0.08
    gray_ok = gray_ratio >= 0.90
    verdict = "PASS" if sat_ok and gray_ok else "FAIL"
    return avg_sat, gray_ratio, verdict


xrays = [
    "data/covid19-radiography-database/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png",
    "data/covid19-radiography-database/COVID-19_Radiography_Dataset/Normal/images/Normal-1.png",
    "data/covid19-radiography-database/COVID-19_Radiography_Dataset/Normal/images/Normal-100.png",
]

colors = [
    ((220, 20,  20),  "red (vivid)"),
    ((20,  180, 20),  "green (vivid)"),
    ((30,  100, 200), "blue (vivid)"),
    ((200, 150, 120), "skin tone"),
    ((140, 120, 110), "dark skin tone"),
    ((180, 160, 140), "warm beige"),
    ((80,  90,  100), "cool grey-blue"),
    ((128, 128, 128), "neutral grey"),
]

print(f"{'Image':<35} {'AvgSat':>8} {'GrayRatio':>10} {'Result':>8}")
print("-" * 68)
print("X-rays:")
for p in xrays:
    sat, gr, v = stats(p)
    print(f"  {p.split('/')[-1]:<33} {sat:>8.4f} {gr:>10.4f} {v:>8}")

print("\nSynthetic colour patches:")
for rgb, name in colors:
    img = Image.new("RGB", (300, 300), rgb)
    buf = io.BytesIO(); img.save(buf, "PNG")
    sat, gr, v = stats(buf.getvalue())
    print(f"  {name:<33} {sat:>8.4f} {gr:>10.4f} {v:>8}")
