"""Phase 1 inspection report: what the robot found, how sure it is, and the evidence, from its own memory only.

Input is one mission result (go2wm.mission.runner.run_mission) and the tile it ran on. The tile is used only to
(a) render the camera evidence images the robot would have taken at the recorded poses and (b) an optional,
clearly separated ground-truth appendix for evaluation. Findings, values, classes and actions come from the
robot's spatial memory.

Per finding:
  class   edge: ADA 303 level change from the measured step (ok <= 6.4 mm, bevel <= 13 mm, else violation)
          depression: same thresholds on the measured depth (a tripping / opening hazard)
          dropout: no depth returns where ground was expected -> possible water or a specular surface, unverified
  status  measured / uncertain / candidate (memory status; candidate = located but never measured)
  action  violation -> repair; bevel -> bevel 1:2; ok -> monitor; uncertain / candidate / dropout -> revisit
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..scene.graph import ADA_LEVEL_BEVEL_M, ADA_LEVEL_OK_M, ada_level_class
from ..sim.terrain import params_with_defaults, sidewalk_height

BASE_Z = 0.33
SEV_ORDER = {"violation": 0, "bevel_required": 1, "unverified": 2, "not_measured": 3, "ok": 4}
SEV_COLOR = {"violation": "#c0392b", "bevel_required": "#e67e22", "unverified": "#2e86c1", "not_measured": "#7f8c8d",
             "ok": "#27ae60"}
ACTION = {"violation": "repair (grind / ramp / replace panel)", "bevel_required": "bevel 1:2",
          "unverified": "revisit with camera", "not_measured": "revisit: measure", "ok": "monitor"}
SENSOR_NAME = {"depth_d405": "D405 (close range)", "depth_d555": "D555", None: "-"}


def classify(f: dict[str, Any]) -> str:
    if f["kind"] == "dropout":
        return "unverified"
    if f.get("value_m") is None:
        return "not_measured"
    return ada_level_class(f["value_m"])


def _pose_at(result: dict[str, Any], t: float) -> np.ndarray:
    traj = np.asarray(result["trajectory"])
    dt = result["time_s"] / max(len(traj) - 1, 1)
    return traj[int(np.clip(round(t / dt), 0, len(traj) - 1))]


def _best_obs(f: dict[str, Any]) -> dict[str, Any]:
    meas = [o for o in f["observations"] if o["value_m"] is not None]
    pool = meas or f["observations"]
    return max(pool, key=lambda o: (o["confidence"], -o["range_m"]))


def _evidence_image(result, terrain, f, ob) -> tuple[np.ndarray, str] | None:
    """Camera crop the robot would have captured at the observation's pose (synthetic RGB)."""
    from ..sim.synthetic_camera import CAMERAS, project, render

    cam_name = "d405_rgb" if ob["sensor"] == "depth_d405" else "d555_rgb"
    cam = CAMERAS[cam_name]
    p = params_with_defaults(terrain)
    x, y, yaw = _pose_at(result, ob["t"])
    base = np.array([x, y, sidewalk_height(x, y, p) + BASE_Z])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    fx, fy = f["xy"]
    r = float(np.clip(0.6 * (f.get("extent_m") or 0.3), 0.15, 0.5)) + 0.12
    corners = np.array([[fx + dx, fy + dy, sidewalk_height(fx + dx, fy + dy, p)] for dx in (-r, r) for dy in (-r, r)])
    u, v = project(cam, corners, base, R)
    if not np.all(np.isfinite(u)):
        return None
    win = (int(max(0, np.floor(u.min()))), int(min(cam["width"], np.ceil(u.max()))),
           int(max(0, np.floor(v.min()))), int(min(cam["height"], np.ceil(v.max()))))
    if win[1] - win[0] < 8 or win[3] - win[2] < 8:
        return None
    step = max(1, int(max(win[1] - win[0], win[3] - win[2]) / 360))  # keep crops <= ~360 px
    if step > 1:  # render at reduced resolution for large crops (whole-frame D405 close-ups)
        cam_small = dict(cam)
        cam_small.update(width=cam["width"] // step, height=cam["height"] // step)
        if "fx" in cam:
            for k in ("fx", "fy", "cx", "cy"):
                cam_small[k] = cam[k] / step
        win = tuple(w // step for w in win)
        CAMERAS["_report_tmp"] = cam_small
        cam_name = "_report_tmp"
    try:
        img, _, _ = render(cam_name, base, R, terrain, window=win, ss=2, seed=int(ob["t"] * 10))
    finally:
        CAMERAS.pop("_report_tmp", None)
    label = f"{'D405' if cam_name == 'd405_rgb' or ob['sensor'] == 'depth_d405' else 'D555'} RGB, t = {ob['t']:.1f} s, range {ob['range_m']:.2f} m"
    return img, label


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _route_map(result, terrain, rows) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    p = params_with_defaults(terrain)
    L, half = p["size"][0], 0.5 * p["width"]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.add_patch(Rectangle((-L / 2, -half), L, 2 * half, color="#e8e6e1", zorder=0))
    tr = np.asarray(result["trajectory"])
    ax.plot(tr[:, 0], tr[:, 1], "-", color="#34495e", lw=1.2, label="robot path")
    for wx, wy, *_ in p.get("route", []):
        ax.plot(wx, wy, "k+", ms=7)
    for r in rows:
        ax.plot(*r["xy"], "o", color=SEV_COLOR[r["class"]], ms=9, mec="k", mew=0.6, zorder=3)
        ax.annotate(r["label"], r["xy"], xytext=(4, 6), textcoords="offset points", fontsize=8)
    for k, col in SEV_COLOR.items():
        ax.plot([], [], "o", color=col, mec="k", mew=0.6, label=k.replace("_", " "))
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-half - 0.4, half + 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("along route [m]")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=7, fontsize=8, frameon=False)
    return _png(fig)


def _img_png(img) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 110, h / 110))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
    ax.axis("off")
    return _png(fig)


def build_report(result: dict[str, Any], terrain: dict[str, Any], title: str = "Sidewalk inspection",
                 evidence: bool = True, gt_appendix: bool = True) -> dict[str, Any]:
    """Returns {"rows": [...], "markdown": str, "html": str}."""
    mem = result["memory"]
    rows = []
    for fid, f in mem["findings"].items():
        cls = classify(f)
        ob = _best_obs(f)
        meas = [o for o in f["observations"] if o["value_m"] is not None]
        rows.append({"id": fid, "kind": f["kind"], "xy": [float(v) for v in f["xy"]], "class": cls,
                     "status": f["status"], "value_mm": None if f.get("value_m") is None else f["value_m"] * 1000,
                     "spread_mm": (f.get("value_spread_m") or 0.0) * 1000, "extent_m": f.get("extent_m"),
                     "sensor": f.get("best_sensor"), "confidence": f.get("confidence", 0.0),
                     "n_det": f.get("n_detections", 0), "n_meas": len(meas), "first_seen_t": f["first_seen_t"],
                     "evidence_obs": ob, "observations": meas[-6:], "action": ACTION[cls]})
    rows.sort(key=lambda r: (SEV_ORDER[r["class"]], r["xy"][0]))
    for k, r in enumerate(rows, 1):
        r["label"] = f"F{k}"
    for r in rows:
        r["image"] = None
        if evidence:
            ev = _evidence_image(result, terrain, mem["findings"][r["id"]], r["evidence_obs"])
            if ev is not None:
                r["image"], r["image_label"] = _img_png(ev[0]), ev[1]
    counts = {c: sum(r["class"] == c for r in rows) for c in SEV_ORDER}
    cov = {s: float(np.mean([g["seen"] for g in segs])) for s, segs in mem["coverage"].items()}
    gt = None
    if gt_appendix:
        from .runner import score_mission

        gt = score_mission(result, terrain)
    md = _markdown(result, rows, counts, cov, title, gt)
    return {"rows": rows, "counts": counts, "coverage": cov, "markdown": md,
            "html": _html(result, terrain, rows, counts, cov, title, gt)}


def _mm(v) -> str:
    return "-" if v is None else f"{abs(v) * 1000:.1f}"


def _val(r) -> str:
    if r["value_mm"] is None:
        return "-"
    unit = "step" if r["kind"] == "edge" else "depth"
    return f"{unit} {abs(r['value_mm']):.1f} mm" + (f" (± {r['spread_mm']:.1f})" if r["spread_mm"] > 0.05 else "")


def _markdown(result, rows, counts, cov, title, gt) -> str:
    L = [f"# {title}", "",
         f"Strategy **{result['policy']}**, sensor profile {result['profile']}; outcome **{result['outcome']}** in "
         f"{result['time_s']:.0f} s over {result['path_m']:.1f} m.", "",
         "Findings: " + ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items() if v) + ".  ",
         "Coverage of the walkway: " + ", ".join(f"{SENSOR_NAME.get(s, s)} {v * 100:.0f} %" for s, v in cov.items()) + ".",
         "", f"ADA 303 thresholds: ok <= {ADA_LEVEL_OK_M * 1000:.1f} mm, bevel <= {ADA_LEVEL_BEVEL_M * 1000:.0f} mm, "
         "above that a violation.", "",
         "| # | type | location x, y [m] | measurement | class | status | best sensor | evidence | action |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['kind']} | {r['xy'][0]:.2f}, {r['xy'][1]:.2f} | {_val(r)} | "
                 f"{r['class'].replace('_', ' ')} | {r['status']} | {SENSOR_NAME.get(r['sensor'], r['sensor'])} | "
                 f"{r['n_det']} detections, {r['n_meas']} measurements | {r['action']} |")
    if gt:
        L += ["", "## Evaluation appendix (ground truth: not available to the robot)", "",
              "| defect | true value [mm] | found | measured [mm] | error [mm] |", "|---|---|---|---|---|"]
        for d in gt["per_defect"]:
            mv = d.get("value_m")
            L.append(f"| {d['gt']} | {abs(d['true_value_m']) * 1000:.1f} | {'yes' if d['found'] else 'no'} | "
                     f"{_mm(mv)} | "
                     f"{_mm(d.get('abs_err_m'))} |")
        L.append(f"\nFalse findings: {gt['false_findings']} of {gt['n_findings']}.")
    return "\n".join(L) + "\n"


def _html(result, terrain, rows, counts, cov, title, gt) -> str:
    e = html.escape
    chips = "".join(f'<span class="chip" style="background:{SEV_COLOR[k]}">{v} {e(k.replace("_", " "))}</span>'
                    for k, v in counts.items() if v)
    cards = []
    for r in rows:
        obs = "".join(f"<li>t {o['t']:.1f} s · {e(SENSOR_NAME.get(o['sensor'], o['sensor']))} at {o['range_m']:.2f} m · "
                      f"{abs(o['value_m']) * 1000:.1f} mm · conf {o['confidence']:.2f}</li>" for o in r["observations"])
        img = (f'<figure><img src="data:image/png;base64,{r["image"]}"><figcaption>{e(r["image_label"])} '
               f'(synthetic)</figcaption></figure>') if r["image"] else '<figure class="noimg">no image in view</figure>'
        cards.append(f"""<section class="card" style="border-left-color:{SEV_COLOR[r['class']]}">
<div class="head"><b>{r['label']}</b> {e(r['kind'])} at x {r['xy'][0]:.2f} m, y {r['xy'][1]:+.2f} m
<span class="chip" style="background:{SEV_COLOR[r['class']]}">{e(r['class'].replace('_', ' '))}</span></div>
<div class="body">{img}<div><p><b>{e(_val(r))}</b> · status {e(r['status'])} · best sensor
{e(SENSOR_NAME.get(r['sensor'], str(r['sensor'])))} · confidence {r['confidence']:.2f}</p>
<p>{r['n_det']} detections, {r['n_meas']} measurements; first seen at t {r['first_seen_t']:.1f} s.</p>
<p><b>Action:</b> {e(r['action'])}</p>{'<ul>' + obs + '</ul>' if obs else ''}</div></div></section>""")
    gt_html = ""
    if gt:
        trs = "".join(f"<tr><td>{e(d['gt'])}</td><td>{abs(d['true_value_m']) * 1000:.1f}</td><td>{'yes' if d['found'] else 'no'}</td>"
                      f"<td>{_mm(d.get('value_m'))}</td>"
                      f"<td>{_mm(d.get('abs_err_m'))}</td></tr>" for d in gt["per_defect"])
        gt_html = (f"<h2>Evaluation appendix</h2><p class='muted'>Ground truth from the simulator; the robot never sees "
                   f"it. False findings: {gt['false_findings']} of {gt['n_findings']}.</p><table><tr><th>defect</th>"
                   f"<th>true [mm]</th><th>found</th><th>measured [mm]</th><th>error [mm]</th></tr>{trs}</table>")
    cov_txt = ", ".join(f"{e(SENSOR_NAME.get(s, s))} {v * 100:.0f} %" for s, v in cov.items())
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title><style>
:root{{--bg:#fafaf8;--fg:#1d2327;--muted:#5f6b73;--card:#fff;--line:#dcdcd6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16191c;--fg:#e6e8ea;--muted:#9aa5ad;--card:#1f2327;--line:#30363b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}}
.chip{{color:#fff;border-radius:10px;padding:1px 9px;margin-right:6px;font-size:13px}}
.card{{background:var(--card);border:1px solid var(--line);border-left:6px solid;border-radius:6px;margin:12px 0;padding:10px 14px}}
.body{{display:flex;gap:16px;flex-wrap:wrap}} figure{{margin:0}} figure img{{max-width:360px;width:100%;border-radius:4px}}
figcaption,.muted{{color:var(--muted);font-size:13px}} .noimg{{color:var(--muted);width:200px}}
img.map{{width:100%}} table{{border-collapse:collapse}} td,th{{border-bottom:1px solid var(--line);padding:3px 10px;text-align:left}}
ul{{padding-left:18px;font-size:13px;color:var(--muted)}}</style></head><body>
<h1>{e(title)}</h1><p>Strategy <b>{e(result['policy'])}</b>, sensor errors <b>{e(result['profile'])}</b>: {e(result['outcome'])} in
{result['time_s']:.0f} s over {result['path_m']:.1f} m. Coverage: {cov_txt}.</p><p>{chips}</p>
<img class="map" src="data:image/png;base64,{_route_map(result, terrain, rows)}">
<p class="muted">ADA 303: ok &le; 6.4 mm, bevel &le; 13 mm, above that a violation. Values are measured by the robot;
images are synthetic renders of what its colour cameras would have captured at the recorded poses.</p>
{''.join(cards) or '<p>No findings.</p>'}{gt_html}</body></html>"""


def save_report(rep: dict[str, Any], out_dir: str | Path, stem: str = "report") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.md").write_text(rep["markdown"], encoding="utf-8")
    (out / f"{stem}.html").write_text(rep["html"], encoding="utf-8")
    slim = [{k: v for k, v in r.items() if k not in ("image", "evidence_obs")} for r in rep["rows"]]
    (out / f"{stem}.json").write_text(json.dumps({"rows": slim, "counts": rep["counts"], "coverage": rep["coverage"]},
                                                 indent=1, default=float))
    return out / f"{stem}.html"
