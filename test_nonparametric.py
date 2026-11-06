"""
Minimal, dependency-light smoke test for the (Non-Parametric) FastMETRO-S-R50 body model.

Unlike the official end2end_inference_bodymesh.py demo, this script does NOT
instantiate the SMPL class (which requires the license-gated
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl file that is not available in this repo),
and does NOT use the opendr/pyrender renderers (not installed in the `filterseg`
conda env on this machine). It only exercises the non-parametric vertex regression
path (backbone -> transformer -> coarse/intermediate/fine mesh vertices) and
visualizes the projected mesh vertices on top of the input image with matplotlib.
"""
import os
import os.path as op
import sys

import numpy as np
import torch
import torchvision.models as tv_models
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_REPO_DIR = op.dirname(op.abspath(__file__))
sys.path.insert(0, _REPO_DIR)
sys.path.insert(0, op.join(_REPO_DIR, "manopth"))

from src.modeling.model import FastMETRO_Body_Network
from src.modeling._mano import get_graph_params, spmm
import src.modeling.data.config as cfg
from src.utils.geometric_layers import orthographic_projection


class MeshSampler(object):
    """Lightweight stand-in for src.modeling._smpl.Mesh that only provides
    `upsample` (the only method FastMETRO_Body_Network.forward() needs), without
    instantiating SMPL() (which requires the license-gated SMPL .pkl file)."""

    def __init__(self, filename, num_downsampling=1, nsize=1, device=torch.device("cuda")):
        self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        self._U = [u.to(device) for u in self._U]
        self._D = [d.to(device) for d in self._D]
        self.num_downsampling = num_downsampling

    def upsample(self, x, n1=1, n2=0):
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x


class Args:
    model_name = "FastMETRO-S"
    model_dim_1 = 512
    model_dim_2 = 128
    feedforward_dim_1 = 2048
    feedforward_dim_2 = 512
    conv_1x1_dim = 2048
    transformer_dropout = 0.1
    transformer_nhead = 8
    pos_type = "sine"
    arch = "resnet50"
    use_smpl_param_regressor = False


def build_model(device):
    backbone = tv_models.resnet50(weights=None)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    mesh_sampler = MeshSampler(filename=cfg.SMPL_sampling_matrix, device=device)
    model = FastMETRO_Body_Network(Args(), backbone, mesh_sampler)

    ckpt_path = op.join("models", "fastmetro_checkpoint", "FastMETRO-S-R50_3dpw_state_dict.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"loaded checkpoint: {ckpt_path}")
    print(f"  missing keys: {len(missing)} (expected: smpl_parameter_regressor.* if any)")
    print(f"  unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_visualize = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def load_fine_mesh_edges():
    """Full-resolution (6890-vertex) SMPL mesh adjacency, so we can draw an actual
    wireframe instead of a loose point cloud -- much easier to judge alignment by eye.
    Loading this raw array (instead of through get_graph_params/Mesh) avoids the
    'M'/'F' entries in the npz, which are pickled with a psbody.mesh class that is
    not installed in this env and would raise on unpickling."""
    data = np.load(cfg.SMPL_sampling_matrix, encoding="latin1", allow_pickle=True)
    adj = data["A"][0].tocoo()
    mask = adj.row < adj.col  # undirected: keep each edge once
    return adj.row[mask], adj.col[mask]


_EDGE_ROW, _EDGE_COL = load_fine_mesh_edges()

# H36M pelvis regressor (plain .npy, not license-gated like the SMPL body-scan .pkl).
# The official demo re-centers predicted vertices on this pelvis joint BEFORE
# projecting/rendering, since pred_cam was regressed assuming pelvis-centered vertices.
_J_REGRESSOR_H36M = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M_correct)).float()
_PELVIS_IDX = 0  # H36M_J17_NAME = ('Pelvis', ...)


def recenter_on_pelvis(vertices):
    device = vertices.device
    regressor = _J_REGRESSOR_H36M.to(device)
    joints_h36m = torch.einsum("bik,ji->bjk", [vertices, regressor])
    pelvis = joints_h36m[:, _PELVIS_IDX, :]
    return vertices - pelvis[:, None, :]


def visualize(img_tensor, vertices_2d, out_path):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    pts = vertices_2d.cpu().numpy()
    # orthographic_projection returns coords in roughly [-1, 1] range over the 224x224 crop
    pts_px = (pts + 1.0) * 0.5 * 224.0

    segments = np.stack([pts_px[_EDGE_ROW], pts_px[_EDGE_COL]], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("input")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].add_collection(LineCollection(segments, colors="red", linewidths=0.4, alpha=0.8))
    axes[1].scatter(pts_px[:, 0], pts_px[:, 1], s=0.5, c="yellow", alpha=0.6)
    axes[1].set_xlim(0, img.shape[1])
    axes[1].set_ylim(img.shape[0], 0)
    axes[1].set_title(f"pred mesh wireframe ({pts_px.shape[0]} verts)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = build_model(device)

    demo_dir = op.join("demo", "human-body")
    out_dir = op.join("demo", "human-body", "nonparametric_test")
    os.makedirs(out_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(demo_dir)
        if f.lower().endswith((".jpg", ".png")) and "pred" not in f
    )

    with torch.no_grad():
        for fname in image_files:
            img_path = op.join(demo_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            img_visual = transform_visualize(img)

            out = model(img_tensor)
            pred_cam = out["pred_cam"]
            pred_vertices_fine = recenter_on_pelvis(out["pred_3d_vertices_fine"])

            print(f"{fname}: pred_cam={pred_cam[0].cpu().numpy()}, "
                  f"vertices_fine shape={tuple(pred_vertices_fine.shape)}")

            vertices_2d = orthographic_projection(pred_vertices_fine, pred_cam)[0]
            out_path = op.join(out_dir, fname[:-4] + "_nonparam_pred.jpg")
            visualize(img_visual, vertices_2d, out_path)

    print("done.")


if __name__ == "__main__":
    main()
