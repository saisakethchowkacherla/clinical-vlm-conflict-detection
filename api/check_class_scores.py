"""Sample DenseNet scores across all four COVID dataset classes."""
import warnings
import numpy as np
import torch
import torchvision.transforms as T
import torchxrayvision as xrv
from PIL import Image

warnings.filterwarnings("ignore")

model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()


def score(path: str) -> dict:
    img = Image.open(path).convert("L")
    arr = xrv.datasets.normalize(np.array(img, dtype=np.float32), maxval=255, reshape=True)
    arr = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])(arr)
    with torch.no_grad():
        out = model(torch.from_numpy(arr).unsqueeze(0))[0]
    return {p: round(float(out[i]), 4) for i, p in enumerate(model.pathologies) if p}


BASE = "data/covid19-radiography-database/COVID-19_Radiography_Dataset"
SAMPLES = {
    "COVID":           [f"{BASE}/COVID/images/COVID-{i}.png" for i in range(1, 6)],
    "Viral Pneumonia": [f"{BASE}/Viral Pneumonia/images/Viral Pneumonia-{i}.png" for i in range(1, 6)],
    "Lung_Opacity":    [f"{BASE}/Lung_Opacity/images/Lung_Opacity-{i}.png" for i in range(1, 6)],
    "Normal":          [f"{BASE}/Normal/images/Normal-{i}.png" for i in range(1, 6)],
}

KEYS = ["Lung Opacity", "Consolidation", "Pneumonia", "Infiltration", "Atelectasis", "Effusion", "Edema"]

header = f"{'Class':<20}" + "".join(f"{k:>14}" for k in KEYS)
print(header)
print("-" * len(header))

for cls, paths in SAMPLES.items():
    scores_list = [score(p) for p in paths]
    avg = {k: round(sum(s.get(k, 0) for s in scores_list) / len(scores_list), 4) for k in KEYS}
    row = f"{'AVG ' + cls:<20}" + "".join(f"{avg[k]:>14.4f}" for k in KEYS)
    print(row)
    for i, s in enumerate(scores_list, 1):
        row = f"  {cls}-{i:<18}" + "".join(f"{s.get(k,0):>14.4f}" for k in KEYS)
        print(row)
    print()
