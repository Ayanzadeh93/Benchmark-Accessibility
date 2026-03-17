# Merge: Floorplan Parsing ↔ Obstacle Detection

The two subsystems are now complete independently. This plan merges them smartly:
- Each can still be run **standalone**
- They can also be run **together**: floorplan first, then obstacle detection with the parsed graph loaded into a live navigation tab

---

## Proposed Changes

### New Files

#### [NEW] launch.py (project root: `c:\Tim\AIDGPT\AidGPT\launch.py`)
Top-level launcher with a simple menu:
```
1. Floorplan Parser  (runs python -m floorparsing ...)
2. Obstacle Detection  (runs python main.py ...)
3. Combined (Floorplan → then Detection with nav tab)
4. Combined (Detection with pre-loaded floorplan from results/)
```
Accepts `--mode` CLI argument for non-interactive launch.

---

### Core: Floorplan State

#### [NEW] [core/floorplan_state.py](file:///c:/Tim/AIDGPT/AidGPT/core/floorplan_state.py)
Lightweight dataclass holding:
- `graph: Optional[FloorPlanGraph]` — the parsed graph
- `nav_steps: List[NavStep]` — all navigation steps
- `current_step_idx: int` — which step the user is on
- `start_room / dest_room: str`
- `source_path: str` — where JSON was loaded from
- `loaded: bool`

Includes `load_from_dir(results_dir)` that reads `graph.json` + `navigation_instructions.json`.

---

### UI: Floorplan Navigation Tab

#### [NEW] [ui/floorplan_nav_tab.py](file:///c:/Tim/AIDGPT/AidGPT/ui/floorplan_nav_tab.py)
Renders a side panel (same width system as [navigation_panel.py](file:///c:/Tim/AIDGPT/AidGPT/ui/navigation_panel.py)) with:

| Section | Content |
|---|---|
| **Header** | "FLOORPLAN NAV" title bar |
| **Route** | `Start → Destination` |
| **Graph Summary** | Node count, edge count, rooms list |
| **Step List** | Numbered steps, current step highlighted in accent color |
| **Current Step** | Large card with action + sensory feedback |
| **Controls** | `[←] [→]` step navigation hint, `'I'` to close |

Uses the same color system as [navigation_panel.py](file:///c:/Tim/AIDGPT/AidGPT/ui/navigation_panel.py) (`config.UI_COLORS`).

---

### Application State

#### [MODIFY] [core/application_state.py](file:///c:/Tim/AIDGPT/AidGPT/core/application_state.py)
Add one field:
```python
self.show_floorplan_tab: bool = False  # toggle with 'I'
```

---

### UI: Input Handler

#### [MODIFY] [ui/input_handler.py](file:///c:/Tim/AIDGPT/AidGPT/ui/input_handler.py)
Add `'i'` key binding:
```python
ord('i'): ('floorplan nav tab', 'show_floorplan_tab'),
```

---

### UI: Help System

#### [MODIFY] [ui/help.py](file:///c:/Tim/AIDGPT/AidGPT/ui/help.py)
Add `I — Floorplan Navigation Tab` to the help text.

---

### Main App

#### [MODIFY] [main.py](file:///c:/Tim/AIDGPT/AidGPT/main.py)

1. **Import** `floorplan_nav_tab` and `FloorplanState`
2. **[initialize_components()](file:///c:/Tim/AIDGPT/AidGPT/main.py#202-280)**: auto-probe `results/graph.json` and load into `FloorplanState`
3. **[_prepare_display_frame()](file:///c:/Tim/AIDGPT/AidGPT/main.py#491-602)**: when `state.show_floorplan_tab` is True, composite the floorplan panel side-by-side (same pattern as nav panel / help panel)
4. **[_resize_window_for_side_by_side()](file:///c:/Tim/AIDGPT/AidGPT/main.py#654-682)**: handle `show_floorplan_tab` case
5. **CLI arg `--floorplan-results`**: optional override for results directory path

---

### Launcher

#### [NEW] [launch.py](file:///c:/Tim/AIDGPT/AidGPT/launch.py)
```python
# Interactive menu + --mode CLI flag
# Mode "floorplan" → launches floorparsing CLI
# Mode "detection" → launches main.py
# Mode "combined" → runs floorparsing, then main.py with --floorplan-results pointing to output
```

---

## Verification Plan

### Manual Tests (step-by-step)

**Test 1: Standalone Floorplan**
```bash
cd c:\Tim\AIDGPT\AidGPT
conda activate IJCAI
python launch.py --mode floorplan --image alt/dataset/floorplan/images/MP-860-3L.png --start "Hall" --dest "Kitchen"
```
Expected: graph extracted, navigation steps printed, `results/graph.json` saved.

**Test 2: Standalone Detection**
```bash
python launch.py --mode detection
```
Expected: normal obstacle detection app launches on webcam.

**Test 3: Floorplan Tab in Running Detection App**
```bash
# First run floorplan standalone (Test 1), then:
python launch.py --mode detection
# Press 'I' key in the running window
```
Expected: floorplan side panel appears with route and steps.

**Test 4: Combined (sequential)**
```bash
python launch.py --mode combined --image alt/dataset/floorplan/images/MP-860-3L.png --start "Hall" --dest "Kitchen"
```
Expected: floorplan runs first in terminal, then detection app opens with nav tab pre-loaded and visible.

**Test 5: Step navigation in tab**
While detection running with floorplan tab open, press `→` and `←` to advance/rewind steps. Verify current step highlights change.

> [!NOTE]
> Tests 1-4 require the IJCAI conda environment and a working camera/video source for the detection steps.
