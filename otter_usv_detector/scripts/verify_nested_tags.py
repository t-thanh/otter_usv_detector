#!/usr/bin/env python3
"""
verify_nested_tags.py — Offline detection check for the nested AprilTag+ArUco board.

Board spec (as described by user):
  Outer: AprilTag DICT_APRILTAG_36h11 ID=10  2000×2000 px
  Inner: ArUco    DICT_4X4_50         ID=1    200×200 px  placed at (x=1000, y=500)

Two key requirements found empirically:
  1. AprilTag 36h11: needs 1 cell (250 px) white quiet zone — generateImageMarker
     does NOT include it.  Detection fails without it.
  2. Inner ArUco: needs a white background patch (≥1 ArUco cell = 33 px per side)
     behind it; placed directly on black AprilTag cells it is invisible to the detector.

Tests performed:
  A  Raw board, no padding                                → expected: BOTH FAIL
  B  +250 px AprilTag border only                         → expected: AprilTag ✓ ArUco ✗
  C  +250 px border, inner ArUco has white background     → expected: BOTH ✓  ← target
  D  AprilTag-only reference (sanity check)               → expected: AprilTag ✓ ArUco ✗

Usage:
  python3 verify_nested_tags.py  /path/to/your_board.png
  python3 verify_nested_tags.py               # synthesises a reference board

Annotated output images are saved next to the input file (or /tmp if no file given).
"""

import sys
import os
import numpy as np
import cv2

# ── Tag parameters ────────────────────────────────────────────────────────────
APRIL_DICT_ID        = cv2.aruco.DICT_APRILTAG_36h11
APRIL_TAG_ID         = 10
APRIL_TOTAL_CELLS    = 8          # 36h11: 6×6 data + 1 black border each side = 8×8
                                  # white quiet zone NOT generated automatically

ARUCO_DICT_ID        = cv2.aruco.DICT_4X4_50
ARUCO_TAG_ID         = 1
ARUCO_TOTAL_CELLS    = 6          # 4×4 data + 1 black border each side = 6×6
                                  # white quiet zone NOT generated automatically

# Positions of inner ArUco in the 2000×2000 board (top-left corner of the marker)
INNER_X = 1000
INNER_Y = 500
INNER_SIZE = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_detector(dict_id):
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin  = 3
    p.adaptiveThreshWinSizeMax  = 53
    p.adaptiveThreshWinSizeStep = 10
    p.minMarkerPerimeterRate    = 0.02
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(dict_id), p)


def _detect(gray, det_april, det_aruco):
    ca, ia, reja = det_april.detectMarkers(gray)
    cu, iu, reju = det_aruco.detectMarkers(gray)
    return {
        'april_ids':     ia.flatten().tolist() if ia is not None else [],
        'april_rej':     len(reja) if reja else 0,
        'april_corners': ca,
        'aruco_ids':     iu.flatten().tolist() if iu is not None else [],
        'aruco_rej':     len(reju) if reju else 0,
        'aruco_corners': cu,
    }


def _add_april_border(img, n_cells=1):
    """Add n_cells × (img_size/APRIL_TOTAL_CELLS) white pixels around the board."""
    cell = img.shape[0] // APRIL_TOTAL_CELLS
    pad  = n_cells * cell
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=255), pad


def _add_aruco_whitepatch(img, x, y, size):
    """Paint a white quiet-zone patch behind the inner ArUco marker (1 ArUco cell)."""
    cell = size // ARUCO_TOTAL_CELLS
    qz   = cell   # 1-cell quiet zone
    out  = img.copy()
    r0   = max(0, y - qz);   r1 = min(img.shape[0], y + size + qz)
    c0   = max(0, x - qz);   c1 = min(img.shape[1], x + size + qz)
    out[r0:r1, c0:c1] = 255
    # Re-paste inner marker (it was overwritten by the white patch)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    small = cv2.aruco.generateImageMarker(aruco_dict, ARUCO_TAG_ID, size)
    out[y:y+size, x:x+size] = small
    return out, qz


def _print_result(label, r):
    ok_a = "✓ DETECTED" if APRIL_TAG_ID in r['april_ids'] else "✗ NOT FOUND"
    ok_u = "✓ DETECTED" if ARUCO_TAG_ID in r['aruco_ids'] else "✗ NOT FOUND"
    print(f"\n  [{label}]")
    print(f"    AprilTag 36h11  ID={APRIL_TAG_ID}: {ok_a}"
          f"  (found: {r['april_ids']}, rejected: {r['april_rej']})")
    print(f"    ArUco   4x4_50  ID={ARUCO_TAG_ID}: {ok_u}"
          f"  (found: {r['aruco_ids']}, rejected: {r['aruco_rej']})")


def _annotate(gray, r, scale=0.4):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if r['april_corners']:
        cv2.aruco.drawDetectedMarkers(
            vis, r['april_corners'],
            np.array([[i] for i in r['april_ids']]) if r['april_ids'] else None,
            (0, 255, 0))
    if r['aruco_corners']:
        cv2.aruco.drawDetectedMarkers(
            vis, r['aruco_corners'],
            np.array([[i] for i in r['aruco_ids']]) if r['aruco_ids'] else None,
            (0, 128, 255))
    h, w = vis.shape[:2]
    if scale != 1.0:
        vis = cv2.resize(vis, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return vis


def _synthesise_board():
    print("[info] No input image — synthesising 2000×2000 reference board.")
    april_dict = cv2.aruco.getPredefinedDictionary(APRIL_DICT_ID)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    board = cv2.aruco.generateImageMarker(april_dict, APRIL_TAG_ID, 2000)
    small = cv2.aruco.generateImageMarker(aruco_dict, ARUCO_TAG_ID, INNER_SIZE)
    board[INNER_Y:INNER_Y+INNER_SIZE, INNER_X:INNER_X+INNER_SIZE] = small
    return board


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: cannot read '{path}'"); sys.exit(1)
        print(f"[info] Loaded '{path}'  {img.shape[1]}×{img.shape[0]} px")
        out_dir = os.path.dirname(os.path.abspath(path))
    else:
        img     = _synthesise_board()
        out_dir = "/tmp"

    h, w = img.shape
    april_cell = min(h, w) // APRIL_TOTAL_CELLS
    aruco_qz   = INNER_SIZE // ARUCO_TOTAL_CELLS    # inner ArUco quiet zone (px)

    print(f"[info] Board:        {w}×{h} px")
    print(f"[info] AprilTag cell: {april_cell} px  →  border needed: ≥{april_cell} px/side")
    print(f"[info] ArUco cell:    {aruco_qz} px  →  white patch needed: ≥{aruco_qz} px/side")
    print(f"[info] Inner ArUco:   top-left ({INNER_X},{INNER_Y}),  size {INNER_SIZE}×{INNER_SIZE} px")

    det_a = _make_detector(APRIL_DICT_ID)
    det_u = _make_detector(ARUCO_DICT_ID)

    # ── Test A: raw ───────────────────────────────────────────────────────────
    rA = _detect(img, det_a, det_u)

    # ── Test B: +1-cell AprilTag border (no ArUco white patch) ───────────────
    img_B, pad_B = _add_april_border(img, n_cells=1)
    rB = _detect(img_B, det_a, det_u)

    # ── Test C: +1-cell AprilTag border + white patch behind inner ArUco ─────
    img_C_base, qz = _add_aruco_whitepatch(img, INNER_X, INNER_Y, INNER_SIZE)
    img_C, pad_C   = _add_april_border(img_C_base, n_cells=1)
    rC = _detect(img_C, det_a, det_u)

    # ── Test D: AprilTag-only reference (no inner ArUco) ─────────────────────
    april_only  = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(APRIL_DICT_ID), APRIL_TAG_ID, min(h, w))
    img_D, _    = _add_april_border(april_only, n_cells=1)
    rD = _detect(img_D, det_a, det_u)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n══════════════ Detection results ══════════════")
    _print_result(f"A  raw board — no padding", rA)
    _print_result(f"B  +{pad_B}px AprilTag border, no ArUco patch", rB)
    _print_result(f"C  +{pad_C}px AprilTag border + {qz}px ArUco white patch  ← FIX", rC)
    _print_result(f"D  AprilTag-only reference (sanity check)", rD)

    # ── Conclusions ───────────────────────────────────────────────────────────
    print("\n══════════════ Conclusions ══════════════")

    # AprilTag padding need
    if not rA['april_ids'] and APRIL_TAG_ID in rB['april_ids']:
        print(f"[1] AprilTag white border: REQUIRED")
        print(f"    Add {april_cell}px white on every side of your 2000×2000 board.")
        print(f"    Final size: {w + 2*april_cell}×{h + 2*april_cell} px")
    elif APRIL_TAG_ID in rA['april_ids']:
        print(f"[1] AprilTag white border: already present in your image ✓")

    # ArUco quiet zone need
    if not rB['aruco_ids'] and ARUCO_TAG_ID in rC['aruco_ids']:
        print(f"[2] Inner ArUco white patch: REQUIRED")
        print(f"    Before compositing, surround the {INNER_SIZE}×{INNER_SIZE} ArUco")
        print(f"    with ≥{qz}px white on every side (1 ArUco cell = {INNER_SIZE}/{ARUCO_TOTAL_CELLS} = {qz}px).")
        print(f"    Paste a {INNER_SIZE + 2*qz}×{INNER_SIZE + 2*qz} white patch at")
        print(f"    ({INNER_X - qz},{INNER_Y - qz}) before pasting the ArUco marker on top.")
    elif ARUCO_TAG_ID in rB['aruco_ids']:
        print(f"[2] Inner ArUco white patch: not needed — already detectable ✓")
    else:
        print(f"[2] Inner ArUco: NOT detected even with white patch — check placement.")

    # Impact of nesting on AprilTag
    if APRIL_TAG_ID in rC['april_ids'] and APRIL_TAG_ID in rD['april_ids']:
        print(f"[3] Nesting does NOT disrupt AprilTag detection ✓")
    elif APRIL_TAG_ID not in rC['april_ids']:
        print(f"[3] WARNING: AprilTag disrupted by embedded ArUco — reconsider placement.")

    # aruco_ros usage note
    print(f"\n[4] aruco_ros usage:")
    print(f"    Both dicts are supported by cv2.aruco (OpenCV {cv2.__version__}).")
    print(f"    aruco_ros runs ONE dictionary per node instance. To detect both:")
    print(f"      node 1: dictionary=DICT_APRILTAG_36h11  marker_id={APRIL_TAG_ID}  size=<physical_m>")
    print(f"      node 2: dictionary=DICT_4X4_50          marker_id={ARUCO_TAG_ID}  size=<physical_m>")

    # ── Save annotated images ─────────────────────────────────────────────────
    print("\n══════════════ Saved images ══════════════")
    scale = min(1.0, 1200 / max(img_C.shape))
    for tag, gray, r in [('A_raw', img, rA), ('B_april_border', img_B, rB),
                           ('C_full_fix', img_C, rC), ('D_april_only', img_D, rD)]:
        vis  = _annotate(gray, r, scale=scale)
        opath = os.path.join(out_dir, f"verify_{tag}.jpg")
        cv2.imwrite(opath, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        status_a = "✓" if APRIL_TAG_ID in r['april_ids'] else "✗"
        status_u = "✓" if ARUCO_TAG_ID in r['aruco_ids'] else "✗"
        print(f"  [{status_a}april {status_u}aruco]  {opath}")


if __name__ == '__main__':
    main()
