"""Offline tests for the DIODE / ETH3D loaders in benchmark/datasets_tier1.py (synthetic files)."""
import os

import cv2
import numpy as np

from benchmark import datasets_tier1 as dt


def _write_diode(root, name="00019_00183_indoors_000_010"):
    d = os.path.join(root, "val", "indoors", "scene_00019", "scan_00183")
    os.makedirs(d)
    rgb = np.full((768, 1024, 3), 120, np.uint8)
    cv2.imwrite(os.path.join(d, name + ".png"), rgb)
    depth = np.full((768, 1024, 1), 3.0, np.float32)
    depth[0, 0, 0] = np.inf
    mask = np.ones((768, 1024), np.float32)
    mask[:10] = 0
    np.save(os.path.join(d, name + "_depth.npy"), depth)
    np.save(os.path.join(d, name + "_depth_mask.npy"), mask)


def test_diode_loader(tmp_path):
    _write_diode(str(tmp_path))
    # an image without depth files is ignored
    cv2.imwrite(os.path.join(str(tmp_path), "val", "indoors", "scene_00019", "scan_00183", "x.png"),
                np.zeros((4, 4, 3), np.uint8))
    s = list(dt.iter_diode(str(tmp_path)))
    assert len(s) == 1
    s = s[0]
    assert s.rgb.shape == (768, 1024, 3) and s.gt.shape == (768, 1024)
    assert not s.valid[:10].any() and s.valid[10:].all()
    assert np.isfinite(s.gt).all()
    assert s.K[0, 0] == 886.81 and s.K[1, 1] == 927.06 and s.K[0, 2] == 512.0
    assert "diode" in dt.LOADERS


def _write_eth3d(root, W=64, H=48):
    sd = os.path.join(root, "office")
    os.makedirs(os.path.join(sd, "images", "dslr_images"))
    os.makedirs(os.path.join(sd, "ground_truth_depth", "dslr_images"))
    os.makedirs(os.path.join(sd, "dslr_calibration_jpg"))
    cv2.imwrite(os.path.join(sd, "images", "dslr_images", "DSC_0001.JPG"),
                np.full((H, W, 3), 90, np.uint8))
    gt = np.full((H, W), 2.5, "<f4")
    gt[:, :8] = np.inf
    gt.tofile(os.path.join(sd, "ground_truth_depth", "dslr_images", "DSC_0001.JPG"))
    with open(os.path.join(sd, "dslr_calibration_jpg", "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        f.write(f"0 THIN_PRISM_FISHEYE {W} {H} 40.0 41.0 31.5 23.5 0 0 0 0 0 0 0 0\n")
    with open(os.path.join(sd, "dslr_calibration_jpg", "images.txt"), "w") as f:
        f.write("# Image list\n")
        f.write("1 1 0 0 0 0 0 0 0 dslr_images/DSC_0001.JPG\n")
        f.write("\n")


def test_eth3d_loader_full_and_downsampled(tmp_path):
    _write_eth3d(str(tmp_path))
    full = list(dt.iter_eth3d(str(tmp_path), down=1))
    assert len(full) == 1 and full[0].gt.shape == (48, 64)
    assert not full[0].valid[:, :8].any() and full[0].valid[:, 8:].all()
    assert full[0].K[0, 0] == 40.0 and full[0].K[1, 2] == 23.5
    d4 = list(dt.iter_eth3d(str(tmp_path), down=4))[0]
    assert d4.gt.shape == (12, 16) and d4.rgb.shape == (12, 16, 3)
    assert np.isclose(d4.K[0, 0], 10.0) and np.isclose(d4.K[0, 2], (31.5 + 0.5) / 4 - 0.5)
    assert not d4.valid[:, :2].any() and d4.valid[:, 2:].all()
    assert np.isfinite(d4.gt).all()


def test_eth3d_downsample_keeps_nearest_sparse_point(tmp_path):
    _write_eth3d(str(tmp_path))
    p = os.path.join(str(tmp_path), "office", "ground_truth_depth", "dslr_images", "DSC_0001.JPG")
    gt = np.full((48, 64), np.inf, "<f4")
    gt[::4, 8::4] = 5.0                 # one sparse point per block from column-block 2 on
    gt[1, 9] = 2.0                      # off-centre, nearer point in block (0, 2) -> wins
    gt.tofile(p)
    s = list(dt.iter_eth3d(str(tmp_path), down=4))[0]
    assert s.gt[0, 2] == 2.0 and s.valid[0, 2]
    assert s.valid[:, 2:].all() and not s.valid[:, :2].any()


def test_eth3d_skips_wrong_size_dump(tmp_path):
    _write_eth3d(str(tmp_path))
    p = os.path.join(str(tmp_path), "office", "ground_truth_depth", "dslr_images", "DSC_0001.JPG")
    np.zeros(10, "<f4").tofile(p)
    assert list(dt.iter_eth3d(str(tmp_path), down=1)) == []
