"""
Flow Performance Studio: regression-model explorer with a saved model library.

    pip install streamlit pandas numpy plotly
    streamlit run tooth_design_app.py

Models are saved to models.json next to this file.
"""
import base64
import hashlib
import hmac
import html
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps

APP_TITLE = "Flow Performance Studio"
DB = Path(__file__).with_name("models.json")
DEFAULT_P = [5, 10, 15, 20, 30, 45, 60, 80]
NAVY, BLUE = "#0B2A5B", "#1F6FEB"
PAL = ["#E67E22", "#2E8B57", "#8E44AD", "#C0392B", "#16A085", "#7F8C8D", "#D4AC0D", "#2980B9"]
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", size=15, color="#0B1F3A")
NUM = r"\d*\.?\d+(?:[eE][-+]?\d+)?"
SEED_EQ = """80 = 13.13 - 0.0369 Toothwidth1@Sketch102 - 1.437 Toothrad1@Sketch102
+ 0.00255 Ang1@Sketch102 - 0.805 Toothrad2@Sketch102 - 0.00595 Ang2@Sketch102
- 0.247 S_ToothH@Sketch102 - 4.555 ToothDepth@Sketch102 - 1.145 L_ToothH@Sketch102"""

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
.stApp p, .stApp label, .stApp input, .stApp textarea, .stApp li, .stApp td, .stApp th, .hero, .brand, h1, h2, h3, h4 {font-family:'Inter','Segoe UI',Arial,sans-serif;}
[data-testid="stIconMaterial"], [class*="material-symbols"], [class*="material-icons"] {font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;}
[data-baseweb="popover"] {z-index:100000 !important;}
div[role="radiogroup"] {flex-wrap:wrap; row-gap:.4rem;}
[data-testid="stExpander"] summary p {font-size:1.15rem; font-weight:800; color:#0B2A5B;}
.stApp {background:#fff; color:#0B1F3A;}
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:#fff; border-bottom:1px solid #D5DEEC;}
.block-container {padding-top:5rem; max-width:1500px;}
[data-testid="stSidebar"] {background:#EEF2F8; border-right:3px solid #0B2A5B;}
[data-testid="stSidebar"][aria-expanded="true"] {min-width:340px;}
.brand {background:#0B2A5B; color:#fff; padding:18px 20px; border-radius:6px; margin-bottom:14px; border-bottom:5px solid #F2A900;}
.brand b {font-size:1.5rem; font-weight:900; line-height:1.15; display:block;}
.brand span {font-size:.85rem; font-weight:600; color:#BFD0EC; letter-spacing:.08em; text-transform:uppercase;}
.hero {background:linear-gradient(120deg,#0B2A5B 0%,#1F4E9E 100%); color:#fff; padding:28px 36px; border-radius:6px; border-left:10px solid #F2A900; margin-bottom:24px;}
.hero h1 {color:#fff; font-size:2.9rem; font-weight:900; margin:0; letter-spacing:-.5px; line-height:1.1;}
.hero p {color:#D6E2F5; font-size:1.2rem; font-weight:500; margin:.5rem 0 0;}
h2, h3, h4 {color:#0B2A5B; font-weight:800;}
h3 {font-size:1.7rem;}
[data-testid="stWidgetLabel"] p {font-size:1.05rem; font-weight:700; color:#0B2A5B;}
[data-testid="stMetric"] {background:#F5F8FD; border:1px solid #D5DEEC; border-top:5px solid #0B2A5B; padding:16px 20px; border-radius:6px;}
[data-testid="stMetricLabel"] p {font-size:1rem; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:#46587A; white-space:normal;}
[data-testid="stMetricValue"] {font-size:clamp(1.8rem,2.3vw,2.6rem); font-weight:900; color:#0B2A5B; white-space:normal; overflow-wrap:anywhere;}
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:3px solid #0B2A5B;}
.stTabs [data-baseweb="tab"] {font-size:1.2rem; font-weight:800; padding:12px 22px; color:#46587A;}
.stTabs [aria-selected="true"] {color:#0B2A5B; background:#E4ECF8;}
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {font-weight:800; font-size:1.05rem; border-radius:4px; padding:.6rem 1.4rem; border:2px solid #0B2A5B;}
.stButton>button[kind^="primary"], .stFormSubmitButton>button[kind^="primary"] {background:#0B2A5B; color:#fff;}
[data-testid="stSidebar"] .stRadio label p {font-size:1.2rem; font-weight:800;}
</style>"""


# ---------- Equations ----------
def parse_equation(text):
    """'80 = 13.1 - 0.04 A@S + ...' (prefix optional, may wrap lines) -> (const, {var: coef}) or None."""
    text = text.replace("\u2212", "-")
    if "=" in text:
        text = text.split("=", 1)[1]
    body = " ".join(text.split())
    c0 = re.match(rf"\s*([-+]?{NUM})", body)
    if not c0:
        return None
    coefs = {}
    for sign, val, name in re.findall(rf"([+-])\s*({NUM})\s+([A-Za-z_][\w@.\[\]]*)", body[c0.end():]):
        coefs[name] = (-1.0 if sign == "-" else 1.0) * float(val)
    return float(c0.group(1)), coefs


def evaluate(eq, values):
    return eq[0] + sum(c * values[v] for v, c in eq[1].items() if v in values)


def short(n):
    return n.split("@")[0]


def pfmt(p):
    return str(int(p)) if float(p).is_integer() else str(p)


def bounds(eqs, rng):
    """Exact min/max of the response over the whole design space (equations are linear)."""
    lows, highs = [], []
    for const, co in eqs.values():
        lows.append(const + sum(min(c * rng[v][0], c * rng[v][1]) for v, c in co.items() if v in rng))
        highs.append(const + sum(max(c * rng[v][0], c * rng[v][1]) for v, c in co.items() if v in rng))
    lo, hi = min(min(lows), 0.0), max(highs)
    return lo, hi + ((hi - lo) * 0.05 or 1.0)


# ---------- Storage ----------
def save_models(m):
    tmp = DB.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=2))
    tmp.replace(DB)


def load_models():
    if DB.exists():
        try:
            return json.loads(DB.read_text())
        except Exception:
            return {}
    seed = {"m_example": {"name": "Example part", "part": "Seeded with the 80 psi equation",
                          "pressure_unit": "psi", "response": "Flow rate", "response_unit": "",
                          "pressures": DEFAULT_P, "equations": {"80": SEED_EQ}, "ranges": {}}}
    save_models(seed)
    return seed


def to_data_uri(f, max_px=1000):
    """Uploaded image -> resized JPEG data URI (kept small so models.json stays light)."""
    img = ImageOps.exif_transpose(Image.open(f))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def img_bytes(uri):
    return base64.b64decode(uri.split(",", 1)[1])


def normalise(raw):
    """Clean an imported JSON object -> ({id: model}, number_skipped). Accepts one model or a dict of models."""
    if isinstance(raw, dict) and "equations" in raw:
        raw = {"m_" + uuid.uuid4().hex[:8]: raw}
    out, skipped = {}, 0
    for mid, m in (raw.items() if isinstance(raw, dict) else []):
        try:
            eqs = {pfmt(float(k)): str(t) for k, t in m["equations"].items()}
            pres = sorted({float(x) for x in m.get("pressures", [])} | {float(k) for k in eqs})
            out[str(mid)] = {"name": str(m["name"]), "part": str(m.get("part", "")),
                             "pressure_unit": str(m.get("pressure_unit", "psi")),
                             "response": str(m.get("response", "Flow rate")),
                             "response_unit": str(m.get("response_unit", "")),
                             "pressures": pres, "equations": eqs,
                             "images": [i for i in m.get("images", []) if isinstance(i, str) and i.startswith("data:image/")],
                             "ranges": {str(v): [float(r[0]), float(r[1])] for v, r in m.get("ranges", {}).items()}}
        except Exception:
            skipped += 1
    return out, skipped


# ---------- UI helpers ----------
def hero(title, sub=""):
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>',
                unsafe_allow_html=True)


def style_fig(fig, h=560):
    fig.update_layout(template="plotly_white", height=h, font=FONT, margin=dict(l=10, r=10, t=70, b=10),
                      legend=dict(font=dict(size=15), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


# ---------- Password gate ----------
PLACEHOLDER_PW = "choose-a-strong-password"


def get_password():
    """Password from .streamlit/secrets.toml (password = "...") or the APP_PASSWORD environment variable."""
    try:
        pw = st.secrets.get("password")
    except Exception:
        pw = None
    pw = pw or os.environ.get("APP_PASSWORD")
    return None if not pw or pw == PLACEHOLDER_PW else str(pw)


def require_login():
    if st.session_state.get("authed"):
        return
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'] {display:none;}</style>",
                unsafe_allow_html=True)
    hero(APP_TITLE, "Sign in to continue")
    pw = get_password()
    if not pw:
        st.error("No password has been set for this app.")
        st.markdown("Create a file called `.streamlit/secrets.toml` next to the app containing "
                    "`password = \"your-password\"` (see `secrets.toml.example`), or set the `APP_PASSWORD` "
                    "environment variable, then restart the app.")
        st.stop()
    left, _ = st.columns([1, 2])
    with left:
        with st.form("login"):
            entry = st.text_input("Password", type="password")
            go_in = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if go_in:
            if hmac.compare_digest(entry.encode(), pw.encode()):
                st.session_state["authed"] = True
                st.rerun()
            time.sleep(1)
            st.error("Incorrect password.")
    st.stop()


# ---------- Model Library page ----------
def library(models):
    hero("Model Library", "Create, edit and save regression models for each part")
    for k, v in st.session_state.pop("_set_state", {}).items():
        st.session_state[k] = v
    if "_goto" in st.session_state:
        st.session_state["lib_sel"] = st.session_state.pop("_goto")
    if "lib_sel" not in st.session_state and models:
        st.session_state["lib_sel"] = list(models)[0]
    if msg := st.session_state.pop("_flash", None):
        st.success(msg)

    sel = st.selectbox("Select a model to edit, or create a new one", ["__new__"] + list(models),
                       key="lib_sel",
                       format_func=lambda i: "+ New model" if i == "__new__" else models[i]["name"])
    if sel == "__new__":
        mid = st.session_state.setdefault("new_mid", "m_" + uuid.uuid4().hex[:8])
        base = {}
    else:
        mid, base = sel, models[sel]

    c1, c2 = st.columns(2)
    name = c1.text_input("Model name", base.get("name", ""), key=f"n_{mid}", placeholder="e.g. Part 4411-B")
    part = c2.text_input("Part description", base.get("part", ""), key=f"pt_{mid}")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    pu = c1.text_input("Pressure unit", base.get("pressure_unit", "psi"), key=f"pu_{mid}")
    rn = c2.text_input("Response name", base.get("response", "Flow rate"), key=f"rn_{mid}")
    ru = c3.text_input("Response unit", base.get("response_unit", ""), key=f"ru_{mid}")
    ptxt = c4.text_input("Pressure levels (comma-separated)",
                         ", ".join(pfmt(p) for p in base.get("pressures", DEFAULT_P)), key=f"pl_{mid}")
    try:
        levels = sorted({float(x) for x in re.split(r"[,\s]+", ptxt.strip()) if x})
    except ValueError:
        st.error("Pressure levels must be numbers separated by commas.")
        return
    if not levels:
        st.error("Enter at least one pressure level.")
        return

    st.markdown("### Part images")
    st.caption("Upload pictures of the part (PNG, JPG or WebP). They appear next to the results when the model is "
               "selected in the Explorer. Images are resized and stored inside the model, so they travel with the "
               "JSON export.")
    ver = st.session_state.get(f"ver_{mid}", 0)
    keep = []
    saved_imgs = base.get("images", [])
    if saved_imgs:
        ic = st.columns(4)
        for i, im in enumerate(saved_imgs):
            with ic[i % 4]:
                st.image(img_bytes(im), use_container_width=True)
                if not st.checkbox("Remove", key=f"rm_{mid}_{ver}_{i}"):
                    keep.append(im)
    ups = st.file_uploader("Add images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
                           key=f"up_{mid}_{ver}")
    new_imgs = []
    for f in ups:
        try:
            new_imgs.append(to_data_uri(f))
        except Exception:
            st.error(f"Couldn't read “{f.name}” as an image.")
    if new_imgs:
        st.caption(f"{len(new_imgs)} new image(s) will be added when you save.")
    images = keep + new_imgs

    st.markdown("### Regression equations")
    with st.expander("Paste equations (optional): one per pressure level", expanded=not base):
        st.caption("Paste Minitab-style equations here, or skip this and build the model in the tables below. "
                   "The 'xx =' prefix is optional. Changing a pasted equation resets the tables below.")
        cols = st.columns(2)
        parsed = {}
        for i, p in enumerate(levels):
            k = pfmt(p)
            with cols[i % 2]:
                t = st.text_area(f"{k} {pu}", base.get("equations", {}).get(k, ""), height=120, key=f"eq_{mid}_{k}")
                if t.strip():
                    r = parse_equation(t)
                    if r and r[1]:
                        parsed[k] = r
                        st.caption(f"✓ {len(r[1])} terms parsed")
                    else:
                        st.caption("✗ could not parse this equation")

    # ----- Editable tables: constants + dimensions -----
    variables = list(dict.fromkeys(v for _, co in parsed.values() for v in co))
    old_r = base.get("ranges", {})
    colname = lambda p: f"{pfmt(p)} {pu}"
    pcols = [colname(p) for p in levels]
    cdf = pd.DataFrame([{colname(p): parsed[pfmt(p)][0] if pfmt(p) in parsed else 0.0 for p in levels}])
    ddf = pd.DataFrame(
        [{"Dimension": v, "Min": old_r.get(v, [0.0, 10.0])[0], "Max": old_r.get(v, [0.0, 10.0])[1],
          **{colname(p): parsed.get(pfmt(p), (0.0, {}))[1].get(v, 0.0) for p in levels}} for v in variables],
        columns=["Dimension", "Min", "Max"] + pcols,
    ).astype({c: float for c in ["Min", "Max"] + pcols})
    sig = hashlib.md5(json.dumps([levels, cdf.to_dict("list"), ddf.to_dict("list")], sort_keys=True).encode()).hexdigest()[:8]
    numcfg = lambda cs: {c: st.column_config.NumberColumn(c, format="%.6g") for c in cs}

    st.markdown("### Constants")
    st.caption("The intercept of the equation at each pressure level.")
    cedit = st.data_editor(cdf, hide_index=True, num_rows="fixed", use_container_width=True,
                           column_config=numcfg(pcols), key=f"ced_{mid}_{sig}")

    st.markdown("### Dimensions")
    st.caption("Each row is a dimension: rename it, set its Min and Max, and enter its coefficient at each pressure "
               "level. Click the + row to add a dimension, or select a row and press Delete to remove one. Row order "
               "sets the slider order in the Explorer. Spaces in names become underscores, and the Explorer shows only "
               "the part before “@”. Blank cells count as 0.")
    dedit = st.data_editor(ddf, num_rows="dynamic", hide_index=True, use_container_width=True,
                           column_config=numcfg(["Min", "Max"] + pcols), key=f"ded_{mid}_{sig}")

    names, rows, errs = [], [], []
    for _, r in dedit.iterrows():
        n = "" if pd.isna(r["Dimension"]) else str(r["Dimension"]).strip().replace(" ", "_")
        if not n:
            continue
        if not re.fullmatch(r"[A-Za-z_][\w@.\[\]]*", n):
            errs.append(f"“{n}” is not a valid name (start with a letter; use letters, numbers, _ . @).")
        elif n in names:
            errs.append(f"Duplicate dimension name “{n}”.")
        else:
            lo = 0.0 if pd.isna(r["Min"]) else float(r["Min"])
            hi = 10.0 if pd.isna(r["Max"]) else float(r["Max"])
            if hi <= lo:
                errs.append(f"{n}: Max must be greater than Min.")
                continue
            names.append(n)
            rows.append((n, lo, hi, {p: 0.0 if pd.isna(r[colname(p)]) else float(r[colname(p)]) for p in levels}))
    if not rows:
        errs.append("Add at least one dimension.")

    eqs_new = {}
    for p in levels:
        k = pfmt(p)
        c0 = cedit.iloc[0][colname(p)]
        c0 = 0.0 if pd.isna(c0) else float(c0)
        if k in parsed or c0 or any(rw[3][p] for rw in rows):
            s = f"{k} = {c0:.10g}"
            for n, lo, hi, cf in rows:
                s += f" {'-' if cf[p] < 0 else '+'} {abs(cf[p]):.10g} {n}"
            eqs_new[k] = s
    if rows and not eqs_new:
        errs.append("Enter at least one equation (a constant or a coefficient).")
    for e in errs:
        st.error(e)

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    if b1.button("Save model", type="primary", disabled=bool(errs), use_container_width=True):
        if not name.strip():
            st.error("Give the model a name.")
        else:
            models[mid] = {"name": name.strip(), "part": part, "pressure_unit": pu, "response": rn,
                           "response_unit": ru, "pressures": levels, "equations": eqs_new,
                           "ranges": {n: [lo, hi] for n, lo, hi, _ in rows}, "images": images}
            save_models(models)
            st.session_state["_goto"] = mid
            st.session_state["_last_model"] = mid
            st.session_state[f"ver_{mid}"] = ver + 1
            st.session_state["_set_state"] = {f"eq_{mid}_{k}": t for k, t in eqs_new.items()}
            st.session_state["_flash"] = f"Saved “{name.strip()}”. Open it in the Explorer."
            st.session_state.pop("new_mid", None)
            st.rerun()
    if sel != "__new__":
        if b2.button("Duplicate", use_container_width=True):
            nid = "m_" + uuid.uuid4().hex[:8]
            models[nid] = {**json.loads(json.dumps(base)), "name": base["name"] + " (copy)"}
            save_models(models)
            st.session_state["_goto"] = nid
            st.session_state["_flash"] = "Model duplicated."
            st.rerun()
        if b3.button("Delete", use_container_width=True) and st.session_state.get(f"cd_{mid}"):
            models.pop(mid)
            save_models(models)
            st.session_state["_goto"] = "__new__"
            st.session_state["_flash"] = "Model deleted."
            st.rerun()
        b3.checkbox("Confirm delete", key=f"cd_{mid}")

    with st.expander("Backup & sharing"):
        st.download_button("Download all models (JSON)", json.dumps(models, indent=2),
                           "models.json", "application/json")
        up = st.file_uploader("Import models from a JSON file", type="json")
        if up and st.button("Import (merge into library)"):
            try:
                new, skipped = normalise(json.loads(up.read()))
                models.update(new)
                save_models(models)
                st.session_state["_flash"] = f"Imported {len(new)} model(s)" + (f"; skipped {skipped} invalid." if skipped else ".")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {e}")


# ---------- Explorer page ----------
def explorer(models):
    if not models:
        hero(APP_TITLE, "No models yet")
        st.info("Open the Model Library to create your first model.")
        return
    ids = list(models)
    if st.session_state.get("exp_model") not in ids:
        st.session_state.pop("exp_model", None)
    last = st.session_state.get("_last_model")
    with st.sidebar:
        mid = st.selectbox("Part / model", ids, index=ids.index(last) if last in ids else 0,
                           key="exp_model", format_func=lambda i: models[i]["name"])
    st.session_state["_last_model"] = mid
    m = models[mid]
    eqs = {}
    for k, t in m["equations"].items():
        r = parse_equation(t)
        if r and r[1]:
            eqs[float(k)] = r
    hero(m["name"], m.get("part") or "Flow rate vs pressure for the selected design")
    if not eqs:
        st.warning("This model has no valid equations. Fix it in the Model Library.")
        return

    P = sorted(eqs)
    pu, rn, ru = m.get("pressure_unit", "psi"), m.get("response", "Flow rate"), m.get("response_unit", "")
    resp = f"{rn} ({ru})" if ru else rn
    variables = list(dict.fromkeys(v for _, co in eqs.values() for v in co))
    rng = {}
    for v in variables:
        lo, hi = m.get("ranges", {}).get(v, [0.0, 10.0])
        rng[v] = (lo, hi if hi > lo else lo + 1.0)

    design = {}
    with st.sidebar:
        st.markdown("#### Design dimensions")
        for v in variables:
            lo, hi = rng[v]
            design[v] = st.slider(short(v), float(lo), float(hi), float((lo + hi) / 2),
                                  step=float((hi - lo) / 200), key=f"d_{mid}_{v}")

    flo, fhi = bounds(eqs, rng)
    with st.expander("Axis controls", expanded=True):
        mode = st.radio("Response axis", ["Auto", "Locked to full design space", "Custom"],
                        horizontal=True, key="ax_mode",
                        help="The pressure axis is always locked to your pressure levels.")
        a2, a3, a4 = st.columns(3)
        cmin = a2.number_input("Custom min", value=round(flo, 2), key=f"cmin_{mid}", disabled=mode != "Custom")
        cmax = a3.number_input("Custom max", value=round(fhi, 2), key=f"cmax_{mid}", disabled=mode != "Custom")
        a4.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        log_x = a4.checkbox("Log pressure axis", key="logx")
    yr = None if mode == "Auto" else ([cmin, cmax] if mode == "Custom" else [flo, fhi])
    rev = f"{mid}|{mode}|{cmin}|{cmax}|{log_x}"

    def curve(d):
        return [evaluate(eqs[p], d) for p in P]

    cur = curve(design)
    imgs = m.get("images", [])
    if imgs:
        ic, kc = st.columns([2, 3])
        with ic:
            i = 0
            if len(imgs) > 1:
                i = st.radio("Image", range(len(imgs)), format_func=lambda j: f"View {j + 1}",
                             horizontal=True, key=f"imgsel_{mid}", label_visibility="collapsed")
            st.image(img_bytes(imgs[i]), use_container_width=True)
    else:
        kc = st.container()
    with kc:
        k1, k2, k3 = st.columns(3)
        k1.metric(f"{rn} @ {pfmt(P[0])} {pu}", f"{cur[0]:,.3f}")
        k2.metric(f"{rn} @ {pfmt(P[-1])} {pu}", f"{cur[-1]:,.3f}")
        k3.metric("Equations loaded", f"{len(P)}")
    st.write("")

    tab_flow, tab_3d, tab_sens = st.tabs(["Flow vs pressure", "3D surface", "Sensitivity"])

    with tab_flow:
        saved = st.session_state.setdefault("saved", {}).setdefault(mid, {})
        st.markdown("**Compare designs:** name the current design, then save it to overlay it on the graph")
        c = st.columns([3, 1, 1])
        sname = c[0].text_input("Design name", placeholder="e.g. Design A", label_visibility="collapsed")
        if c[1].button("Save design", use_container_width=True) and sname.strip():
            saved[sname.strip()] = dict(design)
        if c[2].button("Clear saved", use_container_width=True):
            saved.clear()

        fig = go.Figure()
        for i, (n, d) in enumerate(saved.items()):
            fig.add_trace(go.Scatter(x=P, y=curve(d), mode="lines+markers", name=n,
                                     line=dict(width=3, dash="dash", color=PAL[i % len(PAL)]), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=P, y=cur, mode="lines+markers", name="Current design",
                                 line=dict(width=5, color=NAVY), marker=dict(size=11)))
        logok = log_x and min(P) > 0
        xr = [np.log10(min(P)) - 0.1, np.log10(max(P)) + 0.1] if logok else [0, max(P) * 1.05]
        style_fig(fig)
        fig.update_layout(hovermode="x unified", uirevision=rev)
        fig.update_xaxes(title=dict(text=f"<b>Pressure ({pu})</b>", font=dict(size=18)), tickvals=sorted(set(P[::2] + [P[-1]])) if logok else P, automargin=True,
                         range=xr, type="log" if logok else "linear", gridcolor="#E3E8F0", linecolor=NAVY)
        fig.update_yaxes(title=dict(text=f"<b>{resp}</b>", font=dict(size=18)), range=yr, automargin=True,
                         gridcolor="#E3E8F0", linecolor=NAVY)
        st.plotly_chart(fig, use_container_width=True)
        if min(cur) < 0:
            st.warning("Some predicted values are negative. Check the design is inside the fitted range.")
        table = pd.DataFrame({f"Pressure ({pu})": P, "Current design": cur})
        for n, d in saved.items():
            table[n] = curve(d)
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("Download table (CSV)", table.to_csv(index=False), f"{m['name']}_flow.csv", "text/csv")

    with tab_3d:
        if len(variables) < 2:
            st.warning("Need at least two dimensions.")
        else:
            c1, c2 = st.columns(2)
            xv = c1.selectbox("X axis", variables, format_func=short)
            yv = c2.selectbox("Y axis", [v for v in variables if v != xv], format_func=short)
            show = st.multiselect(f"Pressure levels ({pu})", P, default=[P[-1]], format_func=pfmt)
            xs, ys = np.linspace(*rng[xv], 30), np.linspace(*rng[yv], 30)
            gx, gy = np.meshgrid(xs, ys)
            scales = ["Blues", "Oranges", "Greens", "Purples", "Reds", "Teal", "Greys", "YlOrBr"]
            f3 = go.Figure()
            for i, p in enumerate(show):
                vals = {**design, xv: gx, yv: gy}
                f3.add_trace(go.Surface(x=xs, y=ys, z=evaluate(eqs[p], vals) * np.ones_like(gx),
                                        name=f"{pfmt(p)} {pu}", colorscale=scales[i % len(scales)],
                                        opacity=0.9 if len(show) == 1 else 0.65, showscale=len(show) == 1,
                                        colorbar=dict(title=dict(text=rn, side="right"), len=0.7, thickness=18)))
            if show:
                f3.add_trace(go.Scatter3d(x=[design[xv]], y=[design[yv]], z=[evaluate(eqs[show[-1]], design)],
                                          mode="markers", marker=dict(size=9, color="#D62728"), name="Current"))
            style_fig(f3, 700)
            f3.update_layout(showlegend=len(show) > 1, uirevision=rev + f"|{xv}|{yv}|{show}",
                             scene=dict(xaxis=dict(title=f"<b>{short(xv)}</b>", range=list(rng[xv]), tickfont=dict(size=12)),
                                        yaxis=dict(title=f"<b>{short(yv)}</b>", range=list(rng[yv]), tickfont=dict(size=12)),
                                        zaxis=dict(title=f"<b>{resp}</b>", range=yr, tickfont=dict(size=12)),
                                        aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.7)),
                             scene_camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)))
            st.plotly_chart(f3, use_container_width=True)
            st.caption("Other dimensions are held at the sidebar slider values. X and Y axes are always locked to the design ranges.")

    with tab_sens:
        sp = st.selectbox(f"Pressure level ({pu})", P, index=len(P) - 1, format_func=pfmt)
        co = eqs[sp][1]
        eff = pd.DataFrame({"dimension": [short(v) for v in co],
                            "effect": [c * (rng[v][1] - rng[v][0]) if v in rng else 0 for v, c in co.items()]}
                           ).sort_values("effect", key=abs)
        f2 = go.Figure(go.Bar(x=eff["effect"], y=eff["dimension"], orientation="h",
                              marker_color=["#C0392B" if e < 0 else BLUE for e in eff["effect"]]))
        style_fig(f2, 460)
        mx = max(abs(c) * (rng[v][1] - rng[v][0]) for _, cc in eqs.values() for v, c in cc.items() if v in rng)
        f2.update_xaxes(title=dict(text=f"<b>Change in {rn}, min to max of each dimension</b>", font=dict(size=16)),
                        range=None if mode == "Auto" else [-mx * 1.1, mx * 1.1], gridcolor="#E3E8F0", automargin=True)
        f2.update_yaxes(tickfont=dict(size=16), automargin=True)
        st.plotly_chart(f2, use_container_width=True)
        st.dataframe(pd.DataFrame({"dimension": [short(v) for v in co], "coefficient": list(co.values())}),
                     use_container_width=True, hide_index=True)
        st.write(f"**Constant:** {eqs[sp][0]}")


# ---------- App ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
require_login()
models = load_models()
with st.sidebar:
    st.markdown(f'<div class="brand"><span>Engineering Analytics</span><b>{APP_TITLE}</b></div>',
                unsafe_allow_html=True)
    page = st.radio("Navigation", ["Explorer", "Model Library"], label_visibility="collapsed")
    st.markdown("---")
    if st.button("Log out", use_container_width=True):
        st.session_state["authed"] = False
        st.rerun()
explorer(models) if page == "Explorer" else library(models)
