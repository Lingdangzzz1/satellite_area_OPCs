# app_area.py
# Streamlit app for quantifying + visualizing neuron–OPC “contact” on 2D EM segmentation slices
# pulled from a Neuroglancer segmentation source (via CloudVolume).
#
# Install:
#   pip install streamlit numpy scipy pillow pandas
#   pip install "cloud-volume[gcs]"   # for precomputed://gs://... sources (recommended)
#
# Optional GPU acceleration (for morphology + boolean ops):
#   pip install cupy-cuda12x cupyx  # choose CUDA version that matches your system
#
# Run:
#   streamlit run app_area.py

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.ndimage import binary_dilation as cpu_binary_dilation
from scipy.ndimage import binary_erosion as cpu_binary_erosion

try:
    from cloudvolume import CloudVolume
except Exception:
    CloudVolume = None  # type: ignore


# -----------------------------
# GPU detection + backend ops
# -----------------------------

def _detect_gpu_backend_once() -> Dict[str, Any]:
    """
    Detect a usable CUDA GPU via CuPy. This is intentionally:
      - run once per app open (stored in session_state)
      - independent of Neuroglancer link changes
    """
    info: Dict[str, Any] = {
        "backend": "cpu",
        "gpu_name": None,
        "detail": "CPU mode (NumPy/SciPy).",
    }

    try:
        import cupy as cp  # type: ignore

        try:
            ndev = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            ndev = 0

        if ndev <= 0:
            info["detail"] = "CPU mode (CuPy import succeeded, but no CUDA device detected)."
            return info

        # sanity allocate
        _ = cp.zeros((1,), dtype=cp.uint8)

        # grab device name (best effort)
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", None)
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", errors="ignore")
            info["gpu_name"] = str(name) if name is not None else "CUDA GPU"
        except Exception:
            info["gpu_name"] = "CUDA GPU"

        info["backend"] = "gpu"
        info["detail"] = f"GPU mode enabled via CuPy ({info['gpu_name']})."
        return info

    except Exception as e:
        info["detail"] = f"CPU mode (CuPy not available / unusable): {e}"
        return info


def _get_backend_ops():
    """
    Returns:
      - backend: "cpu" or "gpu"
      - xp: np or cupy module
      - binary_dilation_fn, binary_erosion_fn: ndimage-style functions
      - to_cpu: function to convert arrays back to numpy
    """
    backend_info = st.session_state.get("compute_backend_info", {"backend": "cpu"})
    backend = backend_info.get("backend", "cpu")

    if backend == "gpu":
        try:
            import cupy as cp  # type: ignore
            from cupyx.scipy.ndimage import binary_dilation as gpu_binary_dilation  # type: ignore
            from cupyx.scipy.ndimage import binary_erosion as gpu_binary_erosion  # type: ignore

            def to_cpu(a):
                return cp.asnumpy(a)

            return backend, cp, gpu_binary_dilation, gpu_binary_erosion, to_cpu
        except Exception:
            # fallback to CPU if GPU stack is incomplete at runtime
            backend = "cpu"

    def to_cpu(a):
        return a

    return backend, np, cpu_binary_dilation, cpu_binary_erosion, to_cpu


# -----------------------------
# Output directory persistence
# -----------------------------

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _assert_writable_dir(p: Path) -> Tuple[bool, str]:
    try:
        _safe_mkdir(p)
        test_file = p / ".write_test.tmp"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _run_folder_name(x: int, y: int, zmin: int) -> str:
    # Required naming: folder named as x,y,z (z is smallest z).
    return f"{int(x)}_{int(y)}_{int(zmin)}"


def _ensure_active_run_dir(link_x: int, link_y: int, zmin: int) -> Optional[Path]:
    """
    Create (if needed) and return the active run directory under the user-selected base directory.
    Folder naming rule: <base>/<x>_<y>_<zmin>/
    """
    base_dir_str = st.session_state.get("output_base_dir", "")
    if not base_dir_str:
        return None

    base = Path(base_dir_str).expanduser().resolve()
    folder = _run_folder_name(link_x, link_y, zmin)
    run_dir = base / folder
    _safe_mkdir(run_dir)

    st.session_state.active_run_dir = str(run_dir)
    st.session_state.active_run_zmin = int(zmin)
    st.session_state.active_run_xy = (int(link_x), int(link_y))
    return run_dir


def _save_slice_png(run_dir: Path, link_x: int, link_y: int, z: int, rgba: np.ndarray) -> None:
    """
    Required naming: image files named as x,y,z.
    """
    fname = f"{int(link_x)}_{int(link_y)}_{int(z)}.png"
    out = run_dir / fname
    Image.fromarray(rgba, mode="RGBA").save(out, format="PNG")


def _save_summary_csv(run_dir: Path, df: pd.DataFrame) -> None:
    out = run_dir / "summary.csv"
    df.to_csv(out, index=False)


def _save_metadata_json(run_dir: Path, meta: Dict[str, Any]) -> None:
    out = run_dir / "run_metadata.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _write_zip_of_per_slice_pngs(run_dir: Path) -> None:
    """
    Also persist "Download ZIP of per-slice PNGs" to disk by default.
    Writes: <run_dir>/per_slice_pngs.zip containing all *.png in run_dir.
    """
    pngs = sorted(run_dir.glob("*.png"))
    if len(pngs) == 0:
        return

    zip_path = run_dir / "per_slice_pngs.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pngs:
            # store only the filename inside zip
            zf.write(p, arcname=p.name)


def _list_archived_runs(base_dir: Path) -> List[Path]:
    """
    Archived runs live on disk: folders directly under base_dir that contain summary.csv and/or run_metadata.json.
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    runs: List[Path] = []
    for p in sorted(base_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
        if not p.is_dir():
            continue
        if (p / "summary.csv").exists() or (p / "run_metadata.json").exists():
            runs.append(p)
    return runs


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class LinkInfo:
    raw_url: str
    seg_layer_name: str
    source: str
    selected_ids: List[int]
    x: int
    y: int
    z: int
    link_res_nm: float  # inferred from dimensions['x'][0] * 1e9 (m -> nm), fallback 4.0


@dataclass
class SliceResult:
    ts: float
    source: str
    seg_layer_name: str
    mip: int
    fov: int
    bbox_x0: int
    bbox_y0: int
    z: int
    x_center_link: int
    y_center_link: int
    z_center_link: int
    link_res_nm: float
    vol_res_nm: Tuple[float, float, float]

    present_link_ids: List[int]
    neuron_id: Optional[int]
    opc_ids_kept: List[int]

    neuron_area_px: int
    opc_area_px: int
    contact_area_px: int
    neuron_perimeter_px: int
    ratio_contact_over_perimeter: float

    equivalent_radius_px: float
    contact_start_deg: float
    contact_end_deg: float

    rgba: Optional[np.ndarray] = None
    mask_n: Optional[np.ndarray] = None
    mask_o: Optional[np.ndarray] = None
    mask_c: Optional[np.ndarray] = None


# -----------------------------
# Neuroglancer link parsing
# -----------------------------

def _decode_fragment_to_json(fragment: str) -> Dict[str, Any]:
    """
    Neuroglancer links typically look like:
      https://.../#!%7B%22layers%22%3A...%7D
    Decode the JSON state after "#!" and return it as a dict.
    """
    from urllib.parse import unquote

    frag = fragment
    if frag.startswith("#!"):
        frag = frag[2:]

    # Some links are encoded multiple times
    for _ in range(3):
        frag2 = unquote(frag)
        if frag2 == frag:
            break
        frag = frag2

    frag = frag.strip()

    try:
        return json.loads(frag)
    except json.JSONDecodeError:
        # Fallback: extract a {...} block
        m = re.search(r"(\{.*\})", frag, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


def _first_segmentation_layer(state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    layers = state.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("Neuroglancer state has no 'layers' list.")

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        ltype = str(layer.get("type", "")).lower()
        if "segmentation" in ltype:
            name = str(layer.get("name", "segmentation"))
            return name, layer

    raise ValueError("No segmentation layer found (type contains 'segmentation').")


def _extract_source(layer: Dict[str, Any]) -> str:
    src = layer.get("source")
    if isinstance(src, str):
        return src
    if isinstance(src, dict) and "url" in src:
        return str(src["url"])
    if isinstance(src, list) and len(src) > 0:
        for s in src:
            if isinstance(s, str):
                return s
            if isinstance(s, dict) and "url" in s:
                return str(s["url"])
    raise ValueError("Could not extract CloudVolume source URL from segmentation layer.")


def _extract_selected_ids(layer: Dict[str, Any], state: Dict[str, Any]) -> List[int]:
    """
    Standard pattern: layer['segments'] = ["123", "456"].
    Fallback: scan a few common keys.
    """
    ids: List[int] = []

    segs = layer.get("segments", [])
    if isinstance(segs, list):
        for s in segs:
            try:
                ids.append(int(s))
            except Exception:
                pass

    if not ids:
        for k in ("selectedSegments", "selected_ids", "selection"):
            v = state.get(k)
            if isinstance(v, list):
                for s in v:
                    try:
                        ids.append(int(s))
                    except Exception:
                        pass

    return sorted(set(ids))


def _extract_position_xyz(state: Dict[str, Any]) -> Tuple[int, int, int]:
    pos = state.get("position")
    if isinstance(pos, list) and len(pos) >= 3:
        return int(round(pos[0])), int(round(pos[1])), int(round(pos[2]))

    nav = state.get("navigation", {})
    if isinstance(nav, dict):
        pose = nav.get("pose", {})
        if isinstance(pose, dict):
            ppos = pose.get("position", {})
            if isinstance(ppos, dict):
                vc = ppos.get("voxelCoordinates")
                if isinstance(vc, list) and len(vc) >= 3:
                    return int(round(vc[0])), int(round(vc[1])), int(round(vc[2]))

    raise ValueError("Could not extract viewer position (x,y,z) from Neuroglancer state.")


def _infer_link_resolution_nm(state: Dict[str, Any], default_nm: float = 4.0) -> float:
    dims = state.get("dimensions", {})
    try:
        xdim = dims.get("x", None)
        # Often: dimensions['x'] = [4e-9, 'm']
        if isinstance(xdim, list) and len(xdim) >= 1:
            meters = float(xdim[0])
            nm = meters * 1e9
            if nm > 0:
                return float(nm)
    except Exception:
        pass
    return float(default_nm)


def parse_link_auto(url: str) -> LinkInfo:
    """
    Parse Neuroglancer URL fragment and extract:
      - First segmentation layer
      - Selected segment IDs
      - CloudVolume source
      - Viewer (x,y,z)
      - link_res_nm inferred from dimensions['x'][0] * 1e9; fallback 4.0
    """
    if "#!" not in url:
        raise ValueError("URL does not contain a '#!' Neuroglancer state fragment.")

    fragment = url.split("#!", 1)[1]
    state = _decode_fragment_to_json("#!" + fragment)

    seg_layer_name, seg_layer = _first_segmentation_layer(state)
    source = _extract_source(seg_layer)
    selected_ids = _extract_selected_ids(seg_layer, state)
    x, y, z = _extract_position_xyz(state)
    link_res_nm = _infer_link_resolution_nm(state, default_nm=4.0)

    return LinkInfo(
        raw_url=url,
        seg_layer_name=seg_layer_name,
        source=source,
        selected_ids=selected_ids,
        x=x,
        y=y,
        z=z,
        link_res_nm=link_res_nm,
    )


# -----------------------------
# CloudVolume access + analysis
# -----------------------------

@st.cache_resource(show_spinner=False)
def get_volume(source: str, mip: int) -> Any:
    if CloudVolume is None:
        raise RuntimeError("cloud-volume is not importable. Install with: pip install cloud-volume")
    vol = CloudVolume(
        source,
        mip=mip,
        bounded=True,
        fill_missing=True,
        progress=False,
        parallel=True,
    )
    _ = vol.resolution
    _ = vol.bounds
    return vol


def _bounds_minmax(vol_bounds) -> Optional[Tuple[int, int, int, int, int, int]]:
    try:
        xmin, ymin, zmin = map(int, vol_bounds.minpt)
        xmax, ymax, zmax = map(int, vol_bounds.maxpt)
        return xmin, ymin, zmin, xmax, ymax, zmax
    except Exception:
        return None


def _clamp_bbox_xy(x0: int, x1: int, y0: int, y1: int, vol_bounds) -> Tuple[int, int, int, int]:
    mm = _bounds_minmax(vol_bounds)
    if mm is None:
        return x0, x1, y0, y1
    xmin, ymin, _zmin, xmax, ymax, _zmax = mm
    return max(xmin, x0), min(xmax, x1), max(ymin, y0), min(ymax, y1)


def _link_xyz_to_vol_xyz(link_info: LinkInfo, vol_res_nm: Tuple[float, float, float], x: int, y: int, z: int) -> Tuple[int, int, int]:
    """
    IMPORTANT (matches your reference implementation):
      - Scale X/Y by link_res_nm vs vol.resolution
      - DO NOT rescale Z here (Z is treated as a slice index in the dataset coordinate system)
    """
    sx = link_info.link_res_nm / vol_res_nm[0]
    sy = link_info.link_res_nm / vol_res_nm[1]
    return int(round(x * sx)), int(round(y * sy)), int(z)


def _normalize_cutout_to_2d(cutout: np.ndarray) -> np.ndarray:
    """
    Normalize CloudVolume cutout to a 2D label image (H, W), matching the reference behavior:
      img = cutout[..., 0, 0].T

    CloudVolume commonly returns (X, Y, 1, 1) or (X, Y, 1) etc.
    We:
      1) pick channel indices if present
      2) squeeze to 2D
      3) transpose to match Neuroglancer orientation used in your working code
    """
    arr = np.asarray(cutout)

    # Try to mimic: cutout[..., 0, 0]
    img = arr
    try:
        if img.ndim >= 4:
            img = img[..., 0, 0]
        elif img.ndim == 3:
            img = img[..., 0]
    except Exception:
        # fallback: squeeze later
        img = arr

    img2 = np.squeeze(img)
    if img2.ndim != 2:
        raise ValueError(f"Unexpected cutout shape: {arr.shape} -> {img2.shape}")

    # Match reference orientation
    img2d = img2.T

    if not np.issubdtype(img2d.dtype, np.integer):
        img2d = img2d.astype(np.uint64, copy=False)
    return img2d


def fetch_segmentation_slice(
    vol: Any,
    link_info: LinkInfo,
    z_link: int,
    fov: int,
) -> Tuple[np.ndarray, int, int, int, Tuple[float, float, float]]:
    """
    Download a 2D segmentation cutout at a given Z.

    IMPORTANT (matches your reference implementation):
      - X/Y are scaled from link_res_nm to vol.resolution
      - Z is NOT scaled (treated as slice index)
      - Cutout extraction uses the same channel indexing + transpose convention.
    Returns:
      img2d, bbox_x0, bbox_y0, z_vol, vol_res_nm
    """
    vol_res = tuple(map(float, vol.resolution))  # (nm, nm, nm)

    # Convert link x/y -> vol x/y (scale). Keep z as provided.
    req_x, req_y, zc = _link_xyz_to_vol_xyz(link_info, vol_res, link_info.x, link_info.y, int(z_link))

    half = int(fov) // 2
    x0, x1 = int(req_x - half), int(req_x + half)
    y0, y1 = int(req_y - half), int(req_y + half)
    x0, x1, y0, y1 = _clamp_bbox_xy(x0, x1, y0, y1, vol.bounds)

    # Read one z-plane
    cutout = vol[x0:x1, y0:y1, int(zc):int(zc) + 1]
    img2d = _normalize_cutout_to_2d(cutout)

    return img2d, x0, y0, int(zc), vol_res


def analyze_slice_smart(
    img: np.ndarray,
    link_ids: Sequence[int],
    touch_radius_px: int = 1,
    include_diagonals: bool = True,
) -> Tuple[
    Optional[int], List[int], Dict[int, int],
    np.ndarray, np.ndarray, np.ndarray,
    int, int, int, int, float
]:
    """
    Core computation: how it defines “neuron”, “OPC”, and “contact”
      1) present = link_ids ∩ unique(img)
      2) neuron = largest-area segment among present
      3) OPC candidates = present - {neuron}; keep OPC if dilated(OPC) overlaps neuron
      4) contact area = dilated(OPC_union) ∩ neuron
      5) perimeter = neuron - eroded(neuron) using 3x3 all-neighbors rule
      6) ratio = contact_area / perimeter

    GPU support:
      - If CuPy GPU backend is available (detected at app startup), dilation/erosion + boolean mask ops
        run on GPU; outputs are returned as NumPy arrays to keep the rest of the app unchanged.
    """
    if touch_radius_px < 0:
        touch_radius_px = 0

    backend, xp, nd_dilate, nd_erode, to_cpu = _get_backend_ops()

    unique_ids, counts = np.unique(img, return_counts=True)
    present = sorted(set(int(i) for i in unique_ids) & set(int(i) for i in link_ids))

    h, w = img.shape
    if len(present) == 0:
        mask0 = np.zeros((h, w), dtype=bool)
        return None, [], {}, mask0, mask0, mask0, 0, 0, 0, 0, float("nan")

    present_set = set(present)
    count_map = {int(i): int(c) for i, c in zip(unique_ids.tolist(), counts.tolist()) if int(i) in present_set}

    neuron_id = max(present, key=lambda sid: count_map.get(int(sid), 0))
    opc_candidates = [sid for sid in present if sid != neuron_id]

    mask_n_cpu = (img == neuron_id)

    if include_diagonals:
        structure_cpu = np.ones((3, 3), dtype=bool)
    else:
        structure_cpu = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=bool)

    def dilate(mask_cpu: np.ndarray, r: int) -> np.ndarray:
        if r <= 0:
            return mask_cpu
        if backend == "gpu":
            m = xp.asarray(mask_cpu)
            stc = xp.asarray(structure_cpu)
            out = nd_dilate(m, structure=stc, iterations=int(r))
            return to_cpu(out).astype(bool, copy=False)
        else:
            return cpu_binary_dilation(mask_cpu, structure=structure_cpu, iterations=int(r))

    def erode_all_neighbors(mask_cpu: np.ndarray) -> np.ndarray:
        if backend == "gpu":
            m = xp.asarray(mask_cpu)
            stc = xp.asarray(np.ones((3, 3), dtype=bool))
            out = nd_erode(m, structure=stc, iterations=1)
            return to_cpu(out).astype(bool, copy=False)
        else:
            return cpu_binary_erosion(mask_cpu, structure=np.ones((3, 3), dtype=bool), iterations=1)

    mask_n = mask_n_cpu

    opc_kept: List[int] = []
    opc_masks: List[np.ndarray] = []

    # Optional speed-up on GPU: keep neuron mask on GPU once
    mask_n_gpu = xp.asarray(mask_n_cpu) if backend == "gpu" else None
    stc_gpu = xp.asarray(structure_cpu) if backend == "gpu" else None

    for sid in opc_candidates:
        m_cpu = (img == sid)
        if not m_cpu.any():
            continue

        if touch_radius_px <= 0:
            touches = bool((m_cpu & mask_n).any())
        else:
            if backend == "gpu":
                m = xp.asarray(m_cpu)
                d = nd_dilate(m, structure=stc_gpu, iterations=int(touch_radius_px))
                touches = bool(xp.any(d & mask_n_gpu))
            else:
                touches = bool((dilate(m_cpu, touch_radius_px) & mask_n).any())

        if touches:
            opc_kept.append(int(sid))
            opc_masks.append(m_cpu)

    if len(opc_masks) == 0:
        mask_o = np.zeros_like(mask_n, dtype=bool)
    else:
        mask_o = np.logical_or.reduce(opc_masks)

    mask_c = dilate(mask_o, touch_radius_px) & mask_n

    eroded = erode_all_neighbors(mask_n)
    perimeter = mask_n & (~eroded)

    neuron_area_px = int(mask_n.sum())
    opc_area_px = int(mask_o.sum())
    contact_area_px = int(mask_c.sum())
    perimeter_px = int(perimeter.sum())

    ratio = float(contact_area_px) / float(perimeter_px) if perimeter_px > 0 else float("nan")

    return (
        int(neuron_id),
        opc_kept,
        {int(k): int(v) for k, v in count_map.items()},
        mask_n,
        mask_o,
        mask_c,
        neuron_area_px,
        opc_area_px,
        contact_area_px,
        perimeter_px,
        ratio,
    )


def compute_equivalent_radius_and_contact_arc_deg(
    mask_n: np.ndarray,
    mask_o: np.ndarray,
    include_diagonals: bool = True,
) -> Tuple[float, float, float]:
    """Compute equivalent-circle radius + contact arc start/end angles.

    - Equivalent radius (pixel units): sqrt(area_yellow / pi)
    - Contact set: neuron pixels that touch OPC pixels (4-neighbor, plus diagonals if enabled)
    - Angles: atan2(y, x) around neuron centroid, returned as degrees in [0, 360)
    - Start/End: uses the smallest arc that covers all contact angles (wrap-around handled)
    """

    area = float(np.sum(mask_n))
    if area <= 0:
        return float("nan"), float("nan"), float("nan")

    eq_r = float(np.sqrt(area / np.pi))

    coords_n = np.argwhere(mask_n)
    if coords_n.size == 0:
        return eq_r, float("nan"), float("nan")

    # centroid in array coordinates (row=y, col=x)
    cy = float(coords_n[:, 0].mean())
    cx = float(coords_n[:, 1].mean())

    # Build a neighbor map of OPC pixels (adjacency contacts)
    o = mask_o.astype(bool, copy=False)
    neigh = np.zeros_like(o, dtype=bool)

    # 4-neighbor adjacency
    neigh[:-1, :] |= o[1:, :]
    neigh[1:, :] |= o[:-1, :]
    neigh[:, :-1] |= o[:, 1:]
    neigh[:, 1:] |= o[:, :-1]

    # optional diagonals
    if include_diagonals:
        neigh[:-1, :-1] |= o[1:, 1:]
        neigh[:-1, 1:] |= o[1:, :-1]
        neigh[1:, :-1] |= o[:-1, 1:]
        neigh[1:, 1:] |= o[:-1, :-1]

    contact_px = mask_n & neigh
    coords_c = np.argwhere(contact_px)
    if coords_c.size == 0:
        return eq_r, float("nan"), float("nan")

    # Convert to centered coordinates and compute angles.
    # Note: rows increase downward, so flip sign to get conventional +Y upward.
    dy = -(coords_c[:, 0].astype(np.float64) - cy)
    dx = (coords_c[:, 1].astype(np.float64) - cx)
    ang = np.degrees(np.arctan2(dy, dx))
    ang = (ang + 360.0) % 360.0
    ang.sort()

    if ang.size == 1:
        start = end = float(ang[0])
        return eq_r, start, end

    # Find the largest gap on the circle; the covered arc is its complement.
    diffs = np.diff(ang)
    wrap = float(ang[0] + 360.0 - ang[-1])
    gaps = np.concatenate([diffs, np.array([wrap], dtype=np.float64)])
    k = int(np.argmax(gaps))

    start = float(ang[(k + 1) % ang.size])
    end = float(ang[k])
    return eq_r, start, end


def rgba_overlay(
    mask_n: np.ndarray,
    mask_o: np.ndarray,
    mask_c: Optional[np.ndarray] = None,
    show_contact: bool = True,
) -> np.ndarray:
    """
    Neuron = yellow, OPC = magenta, contact = cyan (optional).
    """
    h, w = mask_n.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Neuron yellow
    rgba[mask_n, 0] = 255
    rgba[mask_n, 1] = 255
    rgba[mask_n, 3] = np.maximum(rgba[mask_n, 3], 255)

    # OPC magenta
    rgba[mask_o, 0] = 255
    rgba[mask_o, 2] = 255
    rgba[mask_o, 3] = np.maximum(rgba[mask_o, 3], 255)

    if show_contact and (mask_c is not None):
        # Contact cyan
        rgba[mask_c, 1] = 255
        rgba[mask_c, 2] = 255
        rgba[mask_c, 3] = np.maximum(rgba[mask_c, 3], 220)

    return rgba


def to_png_bytes(rgba: np.ndarray) -> bytes:
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def summarize_stack(stack: List[SliceResult]) -> pd.DataFrame:
    rows = []
    for s in stack:
        rows.append(
            dict(
                ts=s.ts,
                seg_layer_name=s.seg_layer_name,
                source=s.source,
                mip=s.mip,
                fov=s.fov,
                bbox_x0=s.bbox_x0,
                bbox_y0=s.bbox_y0,
                z=s.z,
                x_center_link=s.x_center_link,
                y_center_link=s.y_center_link,
                z_center_link=s.z_center_link,
                link_res_nm=s.link_res_nm,
                vol_res_nm_x=s.vol_res_nm[0],
                vol_res_nm_y=s.vol_res_nm[1],
                vol_res_nm_z=s.vol_res_nm[2],
                neuron_id=s.neuron_id,
                opc_ids_kept=",".join(map(str, s.opc_ids_kept)),
                present_link_ids=",".join(map(str, s.present_link_ids)),
                neuron_area_px=s.neuron_area_px,
                opc_area_px=s.opc_area_px,
                contact_area_px=s.contact_area_px,
                neuron_perimeter_px=s.neuron_perimeter_px,
                ratio_contact_over_perimeter=s.ratio_contact_over_perimeter,
                equivalent_radius_px=s.equivalent_radius_px,
                contact_start_deg=s.contact_start_deg,
                contact_end_deg=s.contact_end_deg,
            )
        )
    return pd.DataFrame(rows)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Neuron–OPC Contact (2D EM Segmentation)", layout="wide")
st.title("Neuron–OPC “Contact” Quantification on 2D EM Segmentation Slices")

# --- Session init ---
if "compute_backend_info" not in st.session_state:
    st.session_state.compute_backend_info = _detect_gpu_backend_once()

if "output_base_dir" not in st.session_state:
    st.session_state.output_base_dir = ""
if "output_dir_ok" not in st.session_state:
    st.session_state.output_dir_ok = False
if "output_dir_status" not in st.session_state:
    st.session_state.output_dir_status = ""

if "last_url" not in st.session_state:
    st.session_state.last_url = ""
if "link_info" not in st.session_state:
    st.session_state.link_info = None
if "stack" not in st.session_state:
    st.session_state.stack = []
if "acc_n" not in st.session_state:
    st.session_state.acc_n = None
if "acc_o" not in st.session_state:
    st.session_state.acc_o = None
if "acc_c" not in st.session_state:
    st.session_state.acc_c = None
if "last_preview" not in st.session_state:
    st.session_state.last_preview = None

# Disk-backed “archive” pointers (optional; primary archive is on disk)
if "active_run_dir" not in st.session_state:
    st.session_state.active_run_dir = ""
if "active_run_zmin" not in st.session_state:
    st.session_state.active_run_zmin = None
if "active_run_xy" not in st.session_state:
    st.session_state.active_run_xy = None


# --- Startup block (must be BEFORE link paste) ---
with st.container():
    st.subheader("0) Startup (run once per app open): Compute backend + Output directory")

    st.markdown(f"- **Compute**: {st.session_state.compute_backend_info.get('detail', 'CPU mode.')}")
    if st.session_state.compute_backend_info.get("backend") == "gpu":
        st.caption("GPU acceleration is used for morphology + boolean mask ops; CloudVolume I/O remains CPU/network-bound.")

    base_dir_input = st.text_input(
        "Directory to store batch results / archived sessions (external drive path is OK)",
        value=st.session_state.output_base_dir,
        placeholder="/media/<you>/<drive>/neuron_opc_runs",
    ).strip()

    col0a, col0b = st.columns([0.25, 0.75], gap="small")
    with col0a:
        set_dir_btn = st.button("Use directory", type="primary")
    with col0b:
        st.caption("This is required before pasting the Neuroglancer link. Results are saved on disk during batch.")

    if set_dir_btn:
        p = Path(base_dir_input).expanduser()
        ok, msg = _assert_writable_dir(p)
        st.session_state.output_base_dir = str(p.resolve())
        st.session_state.output_dir_ok = bool(ok)
        st.session_state.output_dir_status = msg
        if ok:
            st.success(f"Output directory set: {st.session_state.output_base_dir}")
        else:
            st.error(f"Could not use that directory: {msg}")

    if st.session_state.output_base_dir and st.session_state.output_dir_ok:
        st.markdown(f"- **Active output directory**: `{st.session_state.output_base_dir}`")
    elif st.session_state.output_base_dir and (not st.session_state.output_dir_ok):
        st.warning(f"Output directory not usable: {st.session_state.output_dir_status}")
    else:
        st.info("Set an output directory to enable the rest of the app.")


with st.expander("What this app does (definition of contact, assumptions, caveats)", expanded=False):
    st.markdown(
        """
- **Input**: Paste a Neuroglancer link that contains a **segmentation** layer with a **CloudVolume source URL**, **selected segment IDs**, and viewer **(x,y,z)**.
- **Slice readout**: Downloads a **2D cutout** at a given Z (field-of-view centered on x/y).
- **Role assignment**: Among selected IDs present in the slice, **largest-area segment = neuron**; all others are OPC candidates.
- **Touch filter**: An OPC is kept if it lies within a **1-pixel neighborhood** (dilation) of the neuron.
- **Contact area**: Neuron pixels within **1 pixel** of any kept OPC pixel.
- **Metric**: `ratio = contact_area / neuron_perimeter` (perimeter estimated by erosion-based boundary).
- **Stacking**: Add slices to a stack (or run a batch Z-range). Batch/archived outputs are stored on disk (output directory you set above).
        """
    )


def _finalize_active_run_if_possible(url_for_meta: Optional[str] = None) -> None:
    """
    Write summary.csv + run_metadata.json + per_slice_pngs.zip into the active run directory, if it exists.
    """
    if not st.session_state.output_dir_ok:
        return
    if not st.session_state.active_run_dir:
        return
    run_dir = Path(st.session_state.active_run_dir)
    if not run_dir.exists():
        return
    stack: List[SliceResult] = st.session_state.stack
    if len(stack) == 0:
        return
    df = summarize_stack(stack)
    _save_summary_csv(run_dir, df)
    _write_zip_of_per_slice_pngs(run_dir)

    meta = {
        "saved_at": time.time(),
        "url": url_for_meta or st.session_state.last_url,
        "compute": st.session_state.compute_backend_info,
        "output_dir": st.session_state.output_base_dir,
        "run_dir": str(run_dir),
        "n_slices": len(stack),
    }
    _save_metadata_json(run_dir, meta)


def archive_if_needed(new_url: str) -> None:
    """
    Disk-backed "archive":
      - If the pasted URL changes and there is an active stack, we write summary+metadata+zip to the current run folder,
        then clear the in-memory stack for the next run.
    """
    old_url = st.session_state.last_url
    if old_url and (new_url != old_url) and len(st.session_state.stack) > 0:
        _finalize_active_run_if_possible(url_for_meta=old_url)

        st.session_state.stack = []
        st.session_state.acc_n = None
        st.session_state.acc_o = None
        st.session_state.acc_c = None
        st.session_state.last_preview = None

        # Reset active run pointer (next batch/add will create a new folder)
        st.session_state.active_run_dir = ""
        st.session_state.active_run_zmin = None
        st.session_state.active_run_xy = None


# Disable everything until output directory is set
app_enabled = bool(st.session_state.output_dir_ok and st.session_state.output_base_dir)

left, right = st.columns([1.0, 1.2], gap="large")

with left:
    st.subheader("1) Paste Neuroglancer Link and Parse")
    url = st.text_area(
        "Neuroglancer link (must contain '#!{...}')",
        value=st.session_state.last_url or "",
        height=120,
        placeholder="https://.../#!{...}",
        disabled=not app_enabled,
    ).strip()

    if url and app_enabled:
        archive_if_needed(url)
        st.session_state.last_url = url

    colA, colB = st.columns(2)
    with colA:
        mip = st.number_input("CloudVolume mip", min_value=0, max_value=20, value=0, seeq=1) if False else None
    # The line above is intentionally disabled (legacy); use the real one below.
    colA, colB = st.columns(2)
    with colA:
        mip = st.number_input("CloudVolume mip", min_value=0, max_value=20, value=0, step=1, disabled=not app_enabled)
    with colB:
        # CHANGE 1: default FOV -> 5000
        fov = st.number_input("FOV (pixels)", min_value=64, max_value=8192, value=5000, step=64, disabled=not app_enabled)

    touch_radius = st.number_input("Touch radius (pixels)", min_value=0, max_value=10, value=1, step=1, disabled=not app_enabled)
    include_diagonals = st.checkbox("Include diagonals in touch neighborhood (3×3)", value=True, disabled=not app_enabled)

    parse_btn = st.button("Parse link", type="primary", disabled=(not app_enabled) or (not bool(url)))

    if parse_btn:
        try:
            link_info = parse_link_auto(url)
            st.session_state.link_info = link_info
            st.success("Parsed link successfully.")
        except Exception as e:
            st.session_state.link_info = None
            st.error(f"Failed to parse link: {e}")

    link_info: Optional[LinkInfo] = st.session_state.link_info
    if link_info is not None:
        st.markdown("**Parsed fields**")
        st.json(
            {
                "seg_layer_name": link_info.seg_layer_name,
                "source": link_info.source,
                "selected_ids (count)": len(link_info.selected_ids),
                "selected_ids (first 20)": link_info.selected_ids[:20],
                "position (x,y,z)": (link_info.x, link_info.y, link_info.z),
                "link_res_nm": link_info.link_res_nm,
            }
        )

with right:
    st.subheader("2) Analyze Single Slice or Batch Z-Range")

    link_info = st.session_state.link_info
    if (link_info is None) or (not app_enabled):
        if not app_enabled:
            st.info("Set an output directory above to enable analysis.")
        else:
            st.info("Parse a Neuroglancer link to enable analysis.")
    else:
        mode = st.radio("Mode", ["Single Z slice", "Batch Z-range"], horizontal=True)

        if mode == "Single Z slice":
            z = st.number_input("Z (from link by default)", min_value=0, value=int(link_info.z), step=1)
            run_btn = st.button("Analyze slice (preview)")
            add_btn = st.button("Analyze + Add to Stack")

            def run_analysis(z_val: int, add_to_stack: bool) -> None:
                try:
                    vol = get_volume(link_info.source, mip=int(mip))
                    img2d, x0, y0, zc, vol_res = fetch_segmentation_slice(
                        vol=vol, link_info=link_info, z_link=int(z_val), fov=int(fov)
                    )

                    neuron_id, opc_kept, _area_map, mask_n, mask_o, mask_c, n_area, o_area, c_area, perim, ratio = analyze_slice_smart(
                        img=img2d,
                        link_ids=link_info.selected_ids,
                        touch_radius_px=int(touch_radius),
                        include_diagonals=bool(include_diagonals),
                    )

                    rgba = rgba_overlay(mask_n, mask_o, mask_c, show_contact=True)

                    eq_r, c_start, c_end = compute_equivalent_radius_and_contact_arc_deg(
                        mask_n=mask_n,
                        mask_o=mask_o,
                        include_diagonals=bool(include_diagonals),
                    )

                    present_link_ids = sorted(set(link_info.selected_ids) & set(map(int, np.unique(img2d).tolist())))
                    res = SliceResult(
                        ts=time.time(),
                        source=link_info.source,
                        seg_layer_name=link_info.seg_layer_name,
                        mip=int(mip),
                        fov=int(fov),
                        bbox_x0=int(x0),
                        bbox_y0=int(y0),
                        z=int(z_val),
                        x_center_link=int(link_info.x),
                        y_center_link=int(link_info.y),
                        z_center_link=int(link_info.z),
                        link_res_nm=float(link_info.link_res_nm),
                        vol_res_nm=(float(vol_res[0]), float(vol_res[1]), float(vol_res[2])),
                        present_link_ids=present_link_ids,
                        neuron_id=neuron_id,
                        opc_ids_kept=opc_kept,
                        neuron_area_px=n_area,
                        opc_area_px=o_area,
                        contact_area_px=c_area,
                        neuron_perimeter_px=perim,
                        ratio_contact_over_perimeter=ratio,

                        equivalent_radius_px=float(eq_r),
                        contact_start_deg=float(c_start),
                        contact_end_deg=float(c_end),
                        rgba=rgba,
                        mask_n=mask_n,
                        mask_o=mask_o,
                        mask_c=mask_c,
                    )

                    st.session_state.last_preview = res

                    if add_to_stack:
                        # Active run dir naming uses (x,y,zmin). For single add, zmin == z_val.
                        run_dir = _ensure_active_run_dir(link_info.x, link_info.y, int(z_val))
                        if run_dir is not None:
                            _save_slice_png(run_dir, link_info.x, link_info.y, int(z_val), rgba)
                            # keep run metadata up to date
                            _save_metadata_json(
                                run_dir,
                                {
                                    "url": st.session_state.last_url,
                                    "source": link_info.source,
                                    "seg_layer_name": link_info.seg_layer_name,
                                    "x": int(link_info.x),
                                    "y": int(link_info.y),
                                    "zmin": int(z_val),
                                    "mip": int(mip),
                                    "fov": int(fov),
                                    "touch_radius": int(touch_radius),
                                    "include_diagonals": bool(include_diagonals),
                                    "compute": st.session_state.compute_backend_info,
                                    "saved_at": time.time(),
                                },
                            )

                        if st.session_state.acc_n is None:
                            h, w = mask_n.shape
                            st.session_state.acc_n = np.zeros((h, w), dtype=np.uint16)
                            st.session_state.acc_o = np.zeros((h, w), dtype=np.uint16)
                            st.session_state.acc_c = np.zeros((h, w), dtype=np.uint16)

                        st.session_state.stack.append(res)
                        st.session_state.acc_n += mask_n.astype(np.uint16)
                        st.session_state.acc_o += mask_o.astype(np.uint16)
                        st.session_state.acc_c += mask_c.astype(np.uint16)

                        # CHANGE 2: persist summary.csv + per_slice_pngs.zip to disk by default
                        if run_dir is not None:
                            _save_summary_csv(run_dir, summarize_stack(st.session_state.stack))
                            _write_zip_of_per_slice_pngs(run_dir)

                        st.success(f"Added slice z={z_val} to stack. Stack size: {len(st.session_state.stack)}")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")

            if run_btn:
                run_analysis(int(z), add_to_stack=False)
            if add_btn:
                run_analysis(int(z), add_to_stack=True)

        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                z0 = st.number_input("Z start", min_value=0, value=max(int(link_info.z) - 5, 0), step=1)
            with col2:
                z1 = st.number_input("Z end (inclusive)", min_value=0, value=int(link_info.z) + 5, step=1)
            with col3:
                zstep = st.number_input("Z step", min_value=1, value=1, step=1)

            batch_btn = st.button("Run batch + Add all to Stack", type="primary")

            if batch_btn:
                try:
                    vol = get_volume(link_info.source, mip=int(mip))
                    z_vals = list(range(int(z0), int(z1) + 1, int(zstep)))
                    if len(z_vals) == 0:
                        raise ValueError("Empty Z range.")

                    # Create run folder BEFORE processing (required: folder name uses smallest z)
                    zmin = int(min(z_vals))
                    run_dir = _ensure_active_run_dir(link_info.x, link_info.y, zmin)
                    if run_dir is not None:
                        _save_metadata_json(
                            run_dir,
                            {
                                "url": st.session_state.last_url,
                                "source": link_info.source,
                                "seg_layer_name": link_info.seg_layer_name,
                                "x": int(link_info.x),
                                "y": int(link_info.y),
                                "zmin": int(zmin),
                                "zmax": int(max(z_vals)),
                                "zstep": int(zstep),
                                "mip": int(mip),
                                "fov": int(fov),
                                "touch_radius": int(touch_radius),
                                "include_diagonals": bool(include_diagonals),
                                "compute": st.session_state.compute_backend_info,
                                "started_at": time.time(),
                            },
                        )

                    # First slice sets accumulator shape
                    img2d0, x0, y0, _zc0, _vol_res0 = fetch_segmentation_slice(vol, link_info, z_link=z_vals[0], fov=int(fov))
                    _nid0, _ok0, _am0, mask_n0, mask_o0, mask_c0, *_ = analyze_slice_smart(
                        img2d0, link_info.selected_ids, touch_radius_px=int(touch_radius), include_diagonals=bool(include_diagonals)
                    )
                    h, w = mask_n0.shape
                    if st.session_state.acc_n is None:
                        st.session_state.acc_n = np.zeros((h, w), dtype=np.uint16)
                        st.session_state.acc_o = np.zeros((h, w), dtype=np.uint16)
                        st.session_state.acc_c = np.zeros((h, w), dtype=np.uint16)

                    # UI: only show ONE slice preview at a time during batch
                    preview_slot = st.empty()
                    progress = st.progress(0.0)

                    for idx, z_val in enumerate(z_vals, start=1):
                        img2d, x0, y0, _zc, vol_res = fetch_segmentation_slice(vol, link_info, z_link=z_val, fov=int(fov))

                        neuron_id, opc_kept, _area_map, mask_n, mask_o, mask_c, n_area, o_area, c_area, perim, ratio = analyze_slice_smart(
                            img=img2d,
                            link_ids=link_info.selected_ids,
                            touch_radius_px=int(touch_radius),
                            include_diagonals=bool(include_diagonals),
                        )
                        rgba = rgba_overlay(mask_n, mask_o, mask_c, show_contact=True)

                        eq_r, c_start, c_end = compute_equivalent_radius_and_contact_arc_deg(
                            mask_n=mask_n,
                            mask_o=mask_o,
                            include_diagonals=bool(include_diagonals),
                        )
                        present_link_ids = sorted(set(link_info.selected_ids) & set(map(int, np.unique(img2d).tolist())))

                        res = SliceResult(
                            ts=time.time(),
                            source=link_info.source,
                            seg_layer_name=link_info.seg_layer_name,
                            mip=int(mip),
                            fov=int(fov),
                            bbox_x0=int(x0),
                            bbox_y0=int(y0),
                            z=int(z_val),
                            x_center_link=int(link_info.x),
                            y_center_link=int(link_info.y),
                            z_center_link=int(link_info.z),
                            link_res_nm=float(link_info.link_res_nm),
                            vol_res_nm=(float(vol_res[0]), float(vol_res[1]), float(vol_res[2])),
                            present_link_ids=present_link_ids,
                            neuron_id=neuron_id,
                            opc_ids_kept=opc_kept,
                            neuron_area_px=n_area,
                            opc_area_px=o_area,
                            contact_area_px=c_area,
                            neuron_perimeter_px=perim,
                            ratio_contact_over_perimeter=ratio,

                            equivalent_radius_px=float(eq_r),
                            contact_start_deg=float(c_start),
                            contact_end_deg=float(c_end),
                            rgba=rgba,
                            mask_n=mask_n,
                            mask_o=mask_o,
                            mask_c=mask_c,
                        )

                        st.session_state.stack.append(res)
                        st.session_state.acc_n += mask_n.astype(np.uint16)
                        st.session_state.acc_o += mask_o.astype(np.uint16)
                        st.session_state.acc_c += mask_c.astype(np.uint16)
                        st.session_state.last_preview = res

                        # Save PNG for this slice to disk (required for batch)
                        if run_dir is not None:
                            _save_slice_png(run_dir, link_info.x, link_info.y, int(z_val), rgba)

                        # Update ONE-slice preview (during batch)
                        with preview_slot.container():
                            st.markdown(f"**Batch preview (most recent)** — z={int(z_val)}")
                            cprev1, cprev2 = st.columns([1.1, 1.0], gap="large")
                            with cprev1:
                                st.image(rgba, caption=f"Overlay (z={int(z_val)})", use_container_width=True)
                            with cprev2:
                                st.markdown(
                                    f"""
- **Neuron ID**: `{neuron_id}`
- **Kept OPC IDs**: `{opc_kept}`
- **Neuron area (px)**: `{n_area}`
- **OPC area (px)**: `{o_area}`
- **Contact area (px)**: `{c_area}`
- **Neuron perimeter (px)**: `{perim}`
- **Ratio (contact/perimeter)**: `{ratio:.6g}`
                                    """
                                )

                        progress.progress(idx / len(z_vals))

                    # Persist summary + metadata + zip at end of batch (CHANGE 2)
                    if run_dir is not None:
                        df = summarize_stack(st.session_state.stack)
                        _save_summary_csv(run_dir, df)
                        _write_zip_of_per_slice_pngs(run_dir)
                        _save_metadata_json(
                            run_dir,
                            {
                                "url": st.session_state.last_url,
                                "source": link_info.source,
                                "seg_layer_name": link_info.seg_layer_name,
                                "x": int(link_info.x),
                                "y": int(link_info.y),
                                "zmin": int(zmin),
                                "zmax": int(max(z_vals)),
                                "zstep": int(zstep),
                                "mip": int(mip),
                                "fov": int(fov),
                                "touch_radius": int(touch_radius),
                                "include_diagonals": bool(include_diagonals),
                                "compute": st.session_state.compute_backend_info,
                                "finished_at": time.time(),
                                "n_slices": int(len(z_vals)),
                            },
                        )

                    st.success(f"Batch complete. Added {len(z_vals)} slices. Stack size: {len(st.session_state.stack)}")
                except Exception as e:
                    st.error(f"Batch failed: {e}")

    st.divider()
    st.subheader("3) Preview, Review Results, Downloads (3D temporarily removed)")

    # --- Single-slice preview stays the same (do not change look) ---
    preview: Optional[SliceResult] = st.session_state.last_preview
    if preview is not None:
        st.markdown("**Last preview**")
        c1, c2 = st.columns([1.1, 1.0], gap="large")
        with c1:
            st.image(preview.rgba, caption=f"Overlay (z={preview.z})", use_container_width=True)
        with c2:
            st.markdown(
                f"""
- **Neuron ID**: `{preview.neuron_id}`
- **Kept OPC IDs**: `{preview.opc_ids_kept}`
- **Neuron area (px)**: `{preview.neuron_area_px}`
- **OPC area (px)**: `{preview.opc_area_px}`
- **Contact area (px)**: `{preview.contact_area_px}`
- **Neuron perimeter (px)**: `{preview.neuron_perimeter_px}`
- **Ratio (contact/perimeter)**: `{preview.ratio_contact_over_perimeter:.6g}`
- **Equivalent Radius**: `{preview.equivalent_radius_px:.6g}`
- **Start of Contact**: `{preview.contact_start_deg:.2f}°`
- **End of Contact**: `{preview.contact_end_deg:.2f}°`
                """
            )
            st.download_button(
                "Download preview PNG",
                data=to_png_bytes(preview.rgba),
                file_name=f"preview_{preview.x_center_link}_{preview.y_center_link}_{preview.z}.png",
                mime="image/png",
            )

    # --- Batch / stack review: show only ONE slice at a time ---
    stack: List[SliceResult] = st.session_state.stack
    if len(stack) == 0:
        st.info("No active stack yet. Use “Analyze + Add to Stack” (or batch) to build one.")
    else:
        N = len(stack)
        st.markdown(f"**Active stack size**: {N}")

        if st.session_state.active_run_dir:
            st.markdown(f"- **Saved on disk**: `{st.session_state.active_run_dir}`")
            st.caption("Batch images are saved as x_y_z.png in this folder; summary.csv + per_slice_pngs.zip are written by default.")

        # Pick one slice to view
        default_idx = N - 1
        sel_idx = st.slider("Select slice to view (one at a time)", min_value=0, max_value=N - 1, value=default_idx, step=1)
        s = stack[int(sel_idx)]

        cS1, cS2 = st.columns([1.1, 1.0], gap="large")
        with cS1:
            st.image(s.rgba, caption=f"Overlay (z={s.z})", use_container_width=True)
        with cS2:
            st.markdown(
                f"""
- **Neuron ID**: `{s.neuron_id}`
- **Kept OPC IDs**: `{s.opc_ids_kept}`
- **Neuron area (px)**: `{s.neuron_area_px}`
- **OPC area (px)**: `{s.opc_area_px}`
- **Contact area (px)**: `{s.contact_area_px}`
- **Neuron perimeter (px)**: `{s.neuron_perimeter_px}`
- **Ratio (contact/perimeter)**: `{s.ratio_contact_over_perimeter:.6g}`
- **Equivalent Radius**: `{s.equivalent_radius_px:.6g}`
- **Start of Contact**: `{s.contact_start_deg:.2f}°`
- **End of Contact**: `{s.contact_end_deg:.2f}°`
                """
            )
            st.download_button(
                "Download this slice PNG",
                data=to_png_bytes(s.rgba),
                file_name=f"{s.x_center_link}_{s.y_center_link}_{s.z}.png",
                mime="image/png",
            )

        # Downloads (optional; primary storage is on disk)
        df = summarize_stack(stack)
        colD1, colD2 = st.columns(2)
        with colD1:
            st.download_button(
                "Download CSV summary (current stack)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="slice_summary.csv",
                mime="text/csv",
            )
        with colD2:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for ss in stack:
                    if ss.rgba is None:
                        continue
                    zf.writestr(f"{ss.x_center_link}_{ss.y_center_link}_{ss.z}.png", to_png_bytes(ss.rgba))
            st.download_button(
                "Download ZIP of per-slice PNGs (current stack)",
                data=zbuf.getvalue(),
                file_name="per_slice_pngs.zip",
                mime="application/zip",
            )

        cctrl1, cctrl2 = st.columns(2)
        with cctrl1:
            if st.button("Clear active stack (does not delete disk outputs)"):
                # finalize on clear (best effort)
                _finalize_active_run_if_possible()
                st.session_state.stack = []
                st.session_state.acc_n = None
                st.session_state.acc_o = None
                st.session_state.acc_c = None
                st.session_state.last_preview = None
                st.session_state.active_run_dir = ""
                st.session_state.active_run_zmin = None
                st.session_state.active_run_xy = None
                st.success("Cleared active stack.")
        with cctrl2:
            st.caption("Tip: Changing the pasted URL will finalize summary.csv + per_slice_pngs.zip to disk and reset the active stack automatically.")


st.divider()
st.subheader("Archived Sessions (from disk)")

if not app_enabled:
    st.info("Set an output directory above to view archived sessions on disk.")
else:
    base = Path(st.session_state.output_base_dir).expanduser().resolve()
    runs = _list_archived_runs(base)
    if len(runs) == 0:
        st.info("No archived sessions found in the output directory yet.")
    else:
        # Show only ONE slice preview at a time per archived session
        for i, run_dir in enumerate(runs[:10], start=1):
            with st.expander(f"Archived #{i}: {run_dir.name}", expanded=False):
                st.markdown(f"- **Folder**: `{str(run_dir)}`")

                meta_path = run_dir / "run_metadata.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        st.json(
                            {
                                "x": meta.get("x"),
                                "y": meta.get("y"),
                                "zmin": meta.get("zmin"),
                                "zmax": meta.get("zmax"),
                                "zstep": meta.get("zstep"),
                                "mip": meta.get("mip"),
                                "fov": meta.get("fov"),
                                "touch_radius": meta.get("touch_radius"),
                                "include_diagonals": meta.get("include_diagonals"),
                                "compute": meta.get("compute"),
                            }
                        )
                    except Exception:
                        pass

                pngs = sorted(run_dir.glob("*.png"))
                if len(pngs) == 0:
                    st.write("No PNGs found in this folder.")
                else:
                    # Select one slice image (one at a time)
                    sel = st.selectbox(
                        "Select a slice image to preview (one at a time)",
                        options=[p.name for p in pngs],
                        index=0,
                        key=f"arch_sel_{i}",
                    )
                    img_path = run_dir / sel
                    try:
                        st.image(str(img_path), caption=sel, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display image: {e}")

                    # Downloads
                    st.download_button(
                        "Download this PNG",
                        data=img_path.read_bytes(),
                        file_name=sel,
                        mime="image/png",
                        key=f"dl_png_{i}",
                    )

                csv_path = run_dir / "summary.csv"
                if csv_path.exists():
                    st.download_button(
                        "Download summary.csv",
                        data=csv_path.read_bytes(),
                        file_name=f"{run_dir.name}_summary.csv",
                        mime="text/csv",
                        key=f"dl_csv_{i}",
                    )