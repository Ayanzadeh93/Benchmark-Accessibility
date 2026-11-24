"""Upload all UniDepth student checkpoints to a private Hugging Face repo `depthv2`.

Each run goes in its own folder (best.pt, last.pt, run_info.json holding the training
cfg/epoch/val_absrel saved in the checkpoint). A README.md model card lists every run.

Usage (after `hf auth login`):  python scripts/upload_hf_depthv2.py [--dry-run]
"""
import argparse, json, pathlib, sys
import torch
from huggingface_hub import HfApi

ROOT = pathlib.Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "checkpoints"
STAGE = ROOT / "outputs" / "hf_depthv2_meta"
NOTES = {
    "S21_scale2_full": "INCOMPLETE (died at epoch 5 of 40) — early checkpoint, do not use.",
    "S16_prox": "Main student used for ONNX/int8 edge export and OCMR conformal results.",
    "b10": "B1 5.1M, S16 recipe (s4,s5,ray,range,prox). Best 5M in-domain. IBims d1 66.8, NYU 86.7. ONNX in b10/fp32.onnx.",
    "b20core": "b10 + dsh,cstm,fov,ownar,trim (CanonScale). IBims d1 74.3, NYU 85.6. Same MACs as b10.",
    "b20dws": "RECOMMENDED. b20core + dws4 depthwise-separable stride-4 decoder: 5.04M, 4.04 GMAC (-41%), IBims d1 74.8 / MAE 0.56 m, NYU 85.4, CPU -22..-32% latency. ONNX in b20dws/fp32.onnx.",
    "b20": "b20core + multi-branch rep4. Trained UNSTABLY (scale loss diverged) - do not use; superseded by b20dws.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="depthv2")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--runs", default="", help="comma list: only upload these run folders' weights")
args = ap.parse_args()

STAGE.mkdir(parents=True, exist_ok=True)
rows = []
SKIP = ("_last", "_best", "_smoke")
for run in sorted(p for p in CKPT.iterdir() if (p / "best.pt").exists() and (p / "last.pt").exists() and not p.name.endswith(SKIP)):
    c = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    info = {k: c.get(k) for k in ("epoch", "val_absrel", "cfg")}
    info["note"] = NOTES.get(run.name, "")
    d = STAGE / run.name
    d.mkdir(exist_ok=True)
    (d / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    mb = (run / "best.pt").stat().st_size / 2**20
    va = info["val_absrel"]
    rows.append(f"| `{run.name}` | {info['epoch']} | {va:.4f} | {mb:.0f} MB | {info['note']} |"
                if isinstance(va, float) else f"| `{run.name}` | {info['epoch']} | - | {mb:.0f} MB | {info['note']} |")
    print("staged", run.name, info["epoch"], va)

(STAGE / "README.md").write_text(f"""---
tags: [depth-estimation, knowledge-distillation, unidepth]
---
# depthv2 — UniDepthV2 distilled student checkpoints

Students distilled from UniDepthV2 (ViT-L/B/S teachers). Each folder has `best.pt`
(lowest val AbsRel), `last.pt`, and `run_info.json` (full training config).

Checkpoint format: `torch.load(path, weights_only=False)` -> dict with
`model` (state_dict), `cfg`, `epoch`, `val_absrel`. Load with `distill/` code from the
unidepth2d repo.

Full b10/b20 comparison vs UniDepthV2, MoGe-2, UniK3D: see `B20_RESULTS.md`.

| Run | Best epoch | Val AbsRel | Size | Note |
|---|---|---|---|---|
""" + "\n".join(rows) + "\n", encoding="utf-8")

if args.dry_run:
    print("dry run; staged in", STAGE); sys.exit()

api = HfApi()
user = api.whoami()["name"]
repo = f"{user}/{args.repo}"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=str(STAGE), commit_message="Model card and run info")
runs = [r for r in args.runs.split(",") if r]
pats = [f"{r}/{k}.pt" for r in runs for k in ("best", "last")] if runs else ["*/best.pt", "*/last.pt"]
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=str(CKPT), allow_patterns=pats)
for r in runs:
    onnx = ROOT / "outputs" / "edge" / f"{r}_last" / "fp32.onnx"
    if onnx.exists():
        api.upload_file(path_or_fileobj=str(onnx), path_in_repo=f"{r}/fp32.onnx", repo_id=repo)
res = ROOT / "docs" / "B20_RESULTS.md"
if res.exists():
    api.upload_file(path_or_fileobj=str(res), path_in_repo="B20_RESULTS.md", repo_id=repo)
print("done: https://huggingface.co/" + repo)
