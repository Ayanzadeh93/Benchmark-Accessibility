# Gold measured-range subset — collection protocol

Purpose: a small, real, *measured* egocentric test set (target 200–500 object ranges)
that upgrades OCMR's SUN-derived claims with physically measured ground truth in the
deployment domain (indoor navigation, chest-mounted viewpoint). This is the dataset
nobody else has; it anchors the egocentric/BLV slice of the paper.

## Equipment
- Camera: the deployment camera if possible (otherwise any phone; keep ONE device per
  session). Lock exposure/focus if the app allows. Record device model per session.
- Laser distance meter (preferred, ±2 mm) or tape measure.
- Optional: printed checkerboard (A4, 25 mm squares) — capture 10 frames per device
  once, for intrinsics calibration (`teacher/extract_intrinsics.py` can also estimate).

## Definitions (must be followed exactly)
- **Range** = straight-line distance from the CAMERA'S FRONT PLANE (lens) to the
  NEAREST VISIBLE SURFACE of the object along the aiming line to the point you mark
  in the image. This matches the model-side "visible-surface range" convention
  (median depth in the object region), NOT the object's 3D centre.
- Camera height: measure lens height from floor once per session (chest mount
  ~1.3–1.5 m) and record it — it powers the ground-plane anchoring evaluation (A4).

## Per-object procedure (with the capture tool)
1. Stand still; frame the object; capture (`python -m benchmark.gold_capture live`
   or annotate photos later with `... annotate <folder>`).
2. Measure the distance with the laser at the SAME standing position, aiming at the
   point of the object you will mark. Measure TWICE; if the two differ by >2 cm,
   re-measure. Enter both readings (the tool stores both + their mean).
3. In the tool, drag a tight bbox around the object's VISIBLE extent and click the
   exact aim point; enter class name (use the vocabulary below when it fits).
4. One object may be captured at multiple distances (walk-away series is valuable:
   0.5 / 1 / 2 / 3 / 5 m for the same object).

## Coverage targets (aim, not hard rule)
- Range bands: ≥40 objects in each of 0.3–1 m, 1–2 m, 2–4 m; ≥25 in 4–8 m.
- Classes (BLV-relevant core, ≥10 each): door, chair, table, trash can, person
  (consenting), sink, backpack/bag, sign/label, stairs edge, pole/column.
  Plus free tail — any indoor object counts.
- Conditions: ≥3 rooms; daylight + artificial light; include ≥20 "hard" cases
  (glass door, mirror, black/matte object, strong backlight) and tag them.
- Session hygiene: new session per device/height/room-block (the tool handles ids).

## Storage layout (created by the tool)
```
data_bench/gold/
  sessions.jsonl        # one line per session: device, camera_height_m, lighting, room
  gold_manifest.jsonl   # one line per object: image, bbox, aim_xy, class,
                        # dist1_m, dist2_m, dist_m(mean), session_id, tags
  images/<session>/<n>.jpg
```
Consent note: do not capture identifiable bystanders; people-class captures use a
consenting person only.
