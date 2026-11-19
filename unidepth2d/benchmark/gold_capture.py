"""Gold measured-range capture/annotation tool (see docs/GOLD_CAPTURE_PROTOCOL.md).

Two modes:
  live      webcam session: SPACE captures a frame, then annotate immediately
            python -m benchmark.gold_capture live --camera 0
  annotate  annotate existing photos (e.g., phone shots copied to a folder)
            python -m benchmark.gold_capture annotate <folder>

Annotation flow per object (cv2 window):
  1) drag a tight bbox around the object's visible extent
  2) click the exact aim point you lasered
  3) terminal prompts: class, laser reading 1 (m), reading 2 (m), tags
  keys: u=undo bbox, n=next object same image, ENTER=done image, q=quit session

Appends to <GOLD>/gold_manifest.jsonl; images copied under <GOLD>/images/<session>/.
GOLD = Z:\\Datasets\\OCMR-Gold when the datasets drive is present (canonical, titled
like every other dataset), else the legacy data_bench/gold. Session metadata (device,
camera height, lighting, room) is asked once per run and stored in sessions.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import uuid

import cv2
import numpy as np

from benchmark.datasets_tier1 import DATA_ROOT, DATASETS_ROOT

GOLD = (os.path.join(DATASETS_ROOT, "OCMR-Gold") if os.path.isdir(DATASETS_ROOT)
        else os.path.join(DATA_ROOT, "gold"))


class _BoxUI:
    """Minimal cv2 drag-a-box + click-a-point UI."""

    def __init__(self, win, img):
        self.win, self.base = win, img
        self.box = None          # (x0,y0,x1,y1)
        self.aim = None          # (x,y)
        self._drag = None
        cv2.setMouseCallback(win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag:
            self.box = (*self._drag, x, y)
            self._render()
        elif event == cv2.EVENT_LBUTTONUP and self._drag:
            x0, y0 = self._drag
            self._drag = None
            if abs(x - x0) < 4 and abs(y - y0) < 4:   # click = aim point
                self.aim = (x, y)
            else:
                self.box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            self._render()

    def _render(self):
        img = self.base.copy()
        if self.box:
            x0, y0, x1, y1 = [int(v) for v in self.box]
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if self.aim:
            cv2.drawMarker(img, self.aim, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.imshow(self.win, img)


def _ask(prompt, cast=str, allow_empty=False):
    while True:
        v = input(prompt).strip()
        if not v and allow_empty:
            return None
        try:
            return cast(v)
        except ValueError:
            print("  invalid, retry")


def _new_session(mode: str) -> dict:
    os.makedirs(os.path.join(GOLD, "images"), exist_ok=True)
    sid = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("— new session metadata —")
    meta = {
        "session_id": sid,
        "mode": mode,
        "device": _ask("device (e.g. 'Pixel 8' / 'Logitech C920'): "),
        "camera_height_m": _ask("camera (lens) height above floor in metres: ", float),
        "lighting": _ask("lighting (daylight/artificial/mixed/dim): "),
        "room": _ask("room/location label: "),
        "date": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(GOLD, "sessions.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    os.makedirs(os.path.join(GOLD, "images", sid), exist_ok=True)
    return meta


def _annotate_image(img_path: str, session: dict, manifest_path: str) -> bool:
    """Annotate all objects in one image. Returns False if user quit."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("cannot read", img_path)
        return True
    win = "gold-annotate  [drag box, click aim, n=save object, ENTER=next image, u=undo, q=quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1400, img.shape[1]), min(900, img.shape[0]))
    ui = _BoxUI(win, img)
    cv2.imshow(win, img)
    n_saved = 0
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            cv2.destroyWindow(win)
            return False
        if k == ord("u"):
            ui.box, ui.aim = None, None
            ui._render()
        if k in (13, 10):                       # ENTER -> next image
            break
        if k == ord("n"):
            if not ui.box or not ui.aim:
                print("  need a bbox AND an aim point before 'n'")
                continue
            cls = _ask("  class: ").lower()
            d1 = _ask("  laser reading 1 (m): ", float)
            d2 = _ask("  laser reading 2 (m): ", float)
            if abs(d1 - d2) > 0.02:
                print(f"  !! readings differ by {abs(d1-d2)*100:.1f} cm (>2cm) — flagged")
            tags = _ask("  tags (comma; empty ok, e.g. glass,backlight): ", str, True) or ""
            rec = {
                "id": uuid.uuid4().hex[:10],
                "session_id": session["session_id"],
                "image": os.path.relpath(img_path, GOLD).replace("\\", "/"),
                "bbox_xyxy": [int(v) for v in ui.box],
                "aim_xy": [int(v) for v in ui.aim],
                "class": cls,
                "dist1_m": d1, "dist2_m": d2,
                "dist_m": round((d1 + d2) / 2, 4),
                "readings_disagree": abs(d1 - d2) > 0.02,
                "camera_height_m": session["camera_height_m"],
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
            }
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            n_saved += 1
            print(f"  saved object #{n_saved} ({cls} @ {rec['dist_m']} m)")
            ui.box, ui.aim = None, None
            ui._render()
    cv2.destroyWindow(win)
    return True


def run_live(camera_idx: int):
    session = _new_session("live")
    manifest = os.path.join(GOLD, "gold_manifest.jsonl")
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {camera_idx}")
    win = "gold-capture  [SPACE=capture, q=end session]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    n = 0
    print("SPACE captures a frame; annotate it right away; q ends the session.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            n += 1
            out = os.path.join(GOLD, "images", session["session_id"], f"{n:04d}.jpg")
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 96])
            print(f"captured {out}")
            if not _annotate_image(out, session, manifest):
                break
    cap.release()
    cv2.destroyAllWindows()
    print(f"session {session['session_id']} done: {n} frames")


def run_annotate(folder: str):
    session = _new_session("annotate")
    manifest = os.path.join(GOLD, "gold_manifest.jsonl")
    exts = (".jpg", ".jpeg", ".png", ".webp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    print(f"{len(files)} images in {folder}")
    for f in files:
        src = os.path.join(folder, f)
        dst = os.path.join(GOLD, "images", session["session_id"], f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        print(f"--- {f} ---")
        if not _annotate_image(dst, session, manifest):
            break
    print("annotation session done")


def stats():
    manifest = os.path.join(GOLD, "gold_manifest.jsonl")
    if not os.path.exists(manifest):
        print("no gold data yet")
        return
    recs = [json.loads(l) for l in open(manifest, encoding="utf-8")]
    d = np.array([r["dist_m"] for r in recs])
    from collections import Counter
    print(f"objects={len(recs)} sessions={len(set(r['session_id'] for r in recs))}")
    for lo, hi in ((0.3, 1), (1, 2), (2, 4), (4, 8)):
        print(f"  {lo}-{hi}m: {int(((d>=lo)&(d<hi)).sum())} (target >=40/40/40/25)")
    print("classes:", Counter(r["class"] for r in recs).most_common(15))
    print("flagged (readings disagree):", sum(r.get("readings_disagree", False) for r in recs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_live = sub.add_parser("live")
    p_live.add_argument("--camera", type=int, default=0)
    p_ann = sub.add_parser("annotate")
    p_ann.add_argument("folder")
    sub.add_parser("stats")
    a = ap.parse_args()
    if a.cmd == "live":
        run_live(a.camera)
    elif a.cmd == "annotate":
        run_annotate(a.folder)
    else:
        stats()
