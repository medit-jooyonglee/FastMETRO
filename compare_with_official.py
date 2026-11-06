"""
Direct, apples-to-apples visual comparison for 3dpw_test1.jpg:
  [ input | official repo-shipped render (FastMETRO-L-H64, pyrender) | my wireframe (FastMETRO-S-R50) ]
All three panels are plain 224x224 crops with no matplotlib borders/titles, so the
mesh silhouettes can be compared pixel-for-pixel.
"""
import os.path as op
import numpy as np
import torch
from PIL import Image

import test_nonparametric as tn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tn.build_model(device)

fname = "3dpw_test1.jpg"
img_path = op.join("demo", "human-body", fname)
img = Image.open(img_path).convert("RGB")
img_tensor = tn.transform(img).unsqueeze(0).to(device)
img_visual = tn.transform_visualize(img)

with torch.no_grad():
    out = model(img_tensor)
pred_cam = out["pred_cam"]
pred_vertices_fine = tn.recenter_on_pelvis(out["pred_3d_vertices_fine"])
vertices_2d = tn.orthographic_projection(pred_vertices_fine, pred_cam)[0].cpu().numpy()
pts_px = (vertices_2d + 1.0) * 0.5 * 224.0

# render my wireframe onto a plain 224x224 canvas (no axes/titles/padding)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

img_np = img_visual.permute(1, 2, 0).cpu().numpy()
segments = np.stack([pts_px[tn._EDGE_ROW], pts_px[tn._EDGE_COL]], axis=1)

fig = plt.figure(figsize=(224 / 100, 224 / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img_np)
ax.add_collection(LineCollection(segments, colors="red", linewidths=0.4, alpha=0.9))
ax.set_xlim(0, 224)
ax.set_ylim(224, 0)
ax.axis("off")
fig.savefig("mine_test1_clean.png", dpi=100)
plt.close(fig)

mine_clean = Image.open("mine_test1_clean.png").convert("RGB").resize((224, 224))

official = Image.open(op.join("demo", "human-body", "3dpw_test1_fastmetro_pred.jpg"))
official_input = official.crop((0, 0, 224, 224))
official_render = official.crop((224, 0, 448, 224))

combined = Image.new("RGB", (224 * 3 + 20, 224), "white")
combined.paste(official_input, (0, 0))
combined.paste(official_render, (224 + 10, 0))
combined.paste(mine_clean, (224 * 2 + 20, 0))
combined.save("compare_test1.png")
print("saved compare_test1.png : [input | official FastMETRO-L-H64 render | my FastMETRO-S-R50 wireframe]")
