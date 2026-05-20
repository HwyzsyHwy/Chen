# -*- coding: utf-8 -*-
"""Organic Biomass HTC Multi-Product Forecaster"""
import streamlit as st
import numpy as np
import os
import joblib
import base64
import urllib.request
import pathlib
import pandas as pd

st.set_page_config(page_title="HTC Forecaster", page_icon="🌿",
                   layout="wide", initial_sidebar_state="collapsed")

if "target" not in st.session_state:
    st.session_state.target = "Hydrochar Yield"
if "result" not in st.session_state:
    st.session_state.result = None

# ── 背景图：优先读本地缓存，否则下载并缓存 ──
_BG_URL = "https://raw.githubusercontent.com/HwyzsyHwy/Chen/main/%E8%83%8C%E6%99%AF.png"
_BG_LOCAL = pathlib.Path(__file__).with_name("_bg_cache.png")

@st.cache_data(show_spinner=False)
def _load_bg_base64():
    """返回背景图的 data-URI，本地优先，网络次之"""
    # 1) 本地文件
    if _BG_LOCAL.exists() and _BG_LOCAL.stat().st_size > 1000:
        b64 = base64.b64encode(_BG_LOCAL.read_bytes()).decode()
        return f"data:image/png;base64,{b64}"
    # 2) 尝试多个镜像下载
    urls = [
        _BG_URL,
        "https://ghproxy.net/" + _BG_URL,
        "https://ghfast.top/" + _BG_URL,
        "https://cdn.jsdelivr.net/gh/HwyzsyHwy/Chen@main/%E8%83%8C%E6%99%AF.png",
    ]
    for url in urls:
        try:
            urllib.request.urlretrieve(url, str(_BG_LOCAL))
            if _BG_LOCAL.stat().st_size > 1000:
                b64 = base64.b64encode(_BG_LOCAL.read_bytes()).decode()
                return f"data:image/png;base64,{b64}"
        except Exception:
            continue
    return ""  # 全部失败，返回空

BG = _load_bg_base64()

# ── 训练数据下载 & Type 映射（与模型训练代码完全一致）──
_APP_DIR = pathlib.Path(__file__).parent
_GH_RAW  = "https://raw.githubusercontent.com/HwyzsyHwy/Chen/main/"
_MIRRORS = [_GH_RAW,
            "https://ghfast.top/" + _GH_RAW,
            "https://ghproxy.net/" + _GH_RAW,
            "https://cdn.jsdelivr.net/gh/HwyzsyHwy/Chen@main/"]

# 各目标 → (训练数据文件, 模型文件, 目标列名)
TARGET_CFG = {
    "Hydrochar Yield":   ("HC20260413.xlsx",  "HC_Yield_GBDT_best_model.pkl",  "Yield"),
    "Aqueous phase TN":  ("AP20260413.xlsx",  "AP_TN_GBDT_best_model.pkl",     "TN"),
    "QY of carbon dots": ("CDs20260413.xlsx", "CDs_QY_GBDT_best_model.pkl",    "QY"),
}

def _ensure_file(fname):
    """下载文件到 _APP_DIR，已存在则跳过"""
    local = _APP_DIR / fname
    if local.exists() and local.stat().st_size > 1000:
        return local
    for m in _MIRRORS:
        try:
            urllib.request.urlretrieve(m + fname, str(local))
            if local.stat().st_size > 1000:
                return local
        except Exception:
            continue
    return local

@st.cache_data(show_spinner=False)
def _load_type_info(target):
    """
    返回 (type_list, type_mapping, feature_cols)
    type_list    : 按训练数据首次出现顺序
    type_mapping : {'Food waste': 1, 'Sewage sludge': 2, ...}  从 1 开始
    feature_cols : 特征列名列表（已去掉目标列，含 Type）
    若下载失败/文件损坏，返回 ([], {}, []) 让上层使用 fallback。
    """
    xlsx, _, ycol = TARGET_CFG[target]
    local = _ensure_file(xlsx)
    if not local.exists() or local.stat().st_size < 5000:
        return [], {}, []
    try:
        df = pd.read_excel(str(local))
    except Exception:
        # 文件损坏（很可能是下载到了 HTML 错误页）→ 删除并返回空
        try:
            local.unlink()
        except Exception:
            pass
        return [], {}, []
    if "Type" not in df.columns:
        return [], {}, []
    cats = list(dict.fromkeys(df["Type"]))
    mapping = {c: i + 1 for i, c in enumerate(cats)}
    feat_cols = [c for c in df.columns if c != ycol]
    return cats, mapping, feat_cols

# ────────────────── CSS ──────────────────
st.markdown(f"""<style>
.stApp{{background:#fff!important}}
header[data-testid="stHeader"]{{display:none!important}}
/* 杀掉 stApp 顶部所有间距 */
.stApp > div:first-child{{margin-top:0!important;padding-top:0!important}}
section.main{{padding-top:0!important;margin-top:0!important}}
section.main > div{{padding-top:0!important;margin-top:0!important}}
/* 整体容器宽度 - wide模式下居中限宽 */
section.main .block-container,
div[data-testid="stAppViewBlockContainer"],
div[data-testid="stMainBlockContainer"],
.main .block-container,
[data-testid="stMain"] > div {{
  max-width:1700px!important;
  width:100%!important;
  padding-top:0!important;
  padding-bottom:1.5rem!important;
  padding-left:2rem!important;
  padding-right:2rem!important;
  margin-left:auto!important;
  margin-right:auto!important;
  margin-top:0!important;
}}
/* 确保第一个元素无上边距 */
.block-container > div:first-child{{margin-top:0!important;padding-top:0!important}}
.element-container:first-child{{margin-top:0!important;padding-top:0!important}}
/* 杀掉 stVerticalBlock / stMain 内层所有顶部间距 */
div[data-testid="stVerticalBlock"]{{gap:0!important}}
div[data-testid="stVerticalBlock"] > div:first-child{{margin-top:0!important;padding-top:0!important}}
div[data-testid="stAppViewContainer"]{{padding-top:0!important;margin-top:0!important}}
div[data-testid="stAppViewContainer"] > section{{padding-top:0!important;margin-top:0!important}}
div[data-testid="stMain"]{{padding-top:0!important;margin-top:0!important}}

/* hero - 背景图自适应内容高度 */
.hero{{background:
        linear-gradient(rgba(15,25,55,.22),rgba(15,25,55,.22)),
        url('{BG}') center/cover no-repeat,
        linear-gradient(135deg,#0f1937 0%,#1a2756 50%,#2d2d6e 100%);
       border-radius:14px;
       padding:0px 18px 6px;text-align:center;margin-bottom:36px;
       display:flex;flex-direction:column;
       align-items:center;justify-content:flex-start}}
.hero h1{{font-family:'Times New Roman',Times,serif!important;
          color:#fff!important;font-size:46px!important;font-weight:700!important;
          margin:0 0 8px!important;line-height:1.15!important;
          text-shadow:0 2px 12px rgba(0,0,0,.55);white-space:nowrap}}
.hero p{{font-family:'Times New Roman',Times,serif!important;
         color:#fff!important;font-size:28px!important;line-height:1.45!important;
         margin:0 auto!important;text-shadow:0 1px 6px rgba(0,0,0,.5);
         width:100%!important}}

/* ── 所有 primary / secondary 按钮统一样式 ── */
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-primary"],
button[kind="secondary"],button[kind="primary"]{{
  font-family:'Times New Roman',Times,serif!important;
  font-size:44px!important;font-weight:700!important;color:#fff!important;
  border:none!important;border-radius:10px!important;
  padding:6px 10px!important;min-height:auto!important;
  line-height:1.2!important;
  transition:all .25s ease!important;
}}
/* secondary 按钮内的 p 标签也要改字号 */
button[data-testid="stBaseButton-secondary"] p,
button[data-testid="stBaseButton-primary"] p{{
  font-size:44px!important;font-family:'Times New Roman',Times,serif!important;
  font-weight:700!important;margin:0!important;line-height:1.2!important;
}}
/* 未选中（secondary）= 深蓝 */
button[data-testid="stBaseButton-secondary"]{{
  background:#1b2a4a!important;color:#fff!important;
}}
/* 选中（primary）= 红色 + 阴影 */
button[data-testid="stBaseButton-primary"]{{
  background:#c0392b!important;color:#fff!important;
  box-shadow:0 4px 16px rgba(192,57,43,.35)!important;
}}

/* fieldset look */
.fs{{border:2px solid #c0392b;border-radius:10px;padding:18px 14px 10px;
     position:relative;margin-bottom:10px}}
.fs.org{{border-color:#d4880f}}
.fs.dk{{border-color:#2c3e50}}
.fs.navy{{border-color:#1b2a4a}}
.lg{{position:absolute;top:-11px;left:14px;background:#fff;padding:0 7px;
    font-weight:700;font-size:12.5px;color:#c0392b}}
.lg.org{{color:#d4880f}}
.lg.dk{{color:#2c3e50}}

/* input row */
.irow{{display:flex;align-items:center;margin:5px 0;gap:6px}}
.irow .lab{{min-width:52px;font-weight:700;font-size:13px;color:#333}}
.irow .unit{{font-size:12px;color:#666;min-width:32px}}

/* prediction */
.pred-box{{border:2px solid #1b2a4a;border-radius:12px;padding:22px 18px 18px;
           position:relative;margin-top:14px}}
.pred-lg{{position:absolute;top:-12px;left:50%;transform:translateX(-50%);
          background:#fff;padding:0 10px;font-weight:700;font-size:14px;color:#2c3e50}}
.pred-val{{background:#2d2d5e;color:#fff;border-radius:8px;padding:22px 18px;
           font-size:20px;font-weight:600;min-height:70px;display:flex;align-items:center}}

/* force light theme on all inputs */
[data-testid="stNumberInput"] input{{border:2px solid #e8a030!important;border-radius:6px!important;
  background:#fff!important;color:#222!important}}
[data-testid="stTextInput"] input{{border:2px solid #e8a030!important;border-radius:6px!important;
  background:#fff!important;color:#222!important}}
[data-testid="stSelectbox"] > div > div{{border:2px solid #e8a030!important;border-radius:6px!important;
  background:#fff!important;color:#222!important}}
/* all labels dark */
label, [data-testid="stWidgetLabel"]{{color:#333!important}}

/* ── Target Selection 框和 legend ── */
/* 匹配所有可能的容器标记方式 */
div[data-testid="stHorizontalBlock"][data-ts-styled],
div[data-testid="stHorizontalBlock"].ts-box,
div[data-testid="stHorizontalBlock"].ts-container{{
  border:2px solid #1b2a4a!important;border-radius:12px!important;
  padding:30px 18px 14px!important;position:relative!important;
  margin-bottom:14px!important;overflow:visible!important;
}}
/* 匹配旧JS创建的无class span + 新JS创建的span */
div[data-testid="stHorizontalBlock"][data-ts-styled] > span,
div[data-testid="stHorizontalBlock"].ts-box > span,
div[data-testid="stHorizontalBlock"].ts-container > span{{
  position:absolute!important;top:-18px!important;left:50%!important;
  transform:translateX(-50%)!important;background:#fff!important;
  padding:0 14px!important;font-family:'Times New Roman',Times,serif!important;
  font-weight:700!important;font-size:33px!important;color:#1b2a4a!important;
  white-space:nowrap!important;z-index:10!important;line-height:1.1!important;
}}

/* run / reset — 由 JS 定位 */

/* hide label for compact inputs */
.compact-input label{{display:none!important}}
.compact-input [data-testid="stWidgetLabel"]{{display:none!important}}
</style>""", unsafe_allow_html=True)

# ── JS: 用 components.html 注入可执行脚本 ──
import streamlit.components.v1 as components
import time as _t
_js_ver = str(_t.time())
components.html("""
<script>
/* v=""" + _js_ver + """ */
const P = window.parent.document;

/* ── 断开所有旧的 MutationObserver ── */
if (P._htcObs) { P._htcObs.disconnect(); P._htcObs = null; }

const TGT = ["Hydrochar Yield", "Aqueous phase TN", "QY of carbon dots"];
function applyStyles() {
  const btns = P.querySelectorAll('button');
  let tgtBlock = null;
  btns.forEach(b => {
    const txt = (b.textContent || '').trim();
    if (TGT.includes(txt)) {
      if (!tgtBlock) {
        let el = b;
        while (el && el !== P.body) {
          if (el.getAttribute && el.getAttribute('data-testid') === 'stHorizontalBlock') {
            tgtBlock = el; break;
          }
          el = el.parentElement;
        }
        if (tgtBlock) {
          /* 只加类名，框+legend全靠CSS ::before 控制 */
          tgtBlock.classList.add('ts-box');
          /* 删除所有旧的JS创建的legend span */
          tgtBlock.querySelectorAll('span').forEach(s => {
            if ((s.textContent || '').trim() === 'Target Selection') s.remove();
          });
        }
      }
      /* ── 按钮样式 ── */
      const kind = b.getAttribute('kind');
      Object.assign(b.style, {
        fontFamily: "'Times New Roman', Times, serif",
        fontSize: "44px", fontWeight: "700", color: "#fff",
        border: "none", borderRadius: "10px",
        padding: "6px 10px", minHeight: "auto", lineHeight: "1.2",
        transition: "all .25s ease", width: "100%"
      });
      const p = b.querySelector('p');
      if (p) { p.style.fontSize = "44px"; p.style.fontFamily = "'Times New Roman', Times, serif"; p.style.fontWeight = "700"; p.style.margin = "0"; p.style.lineHeight = "1.2"; }
      if (kind === "primary") {
        b.style.background = "#c0392b";
        b.style.boxShadow = "0 4px 16px rgba(192,57,43,.35)";
      } else {
        b.style.background = "#1b2a4a";
        b.style.boxShadow = "none";
      }
    }
    if (txt === "Run Prediction") {
      Object.assign(b.style, {
        background: "#3a8a8a", color: "#fff", border: "none",
        borderRadius: "8px", fontSize: "15px", fontWeight: "700",
        padding: "12px", width: "100%", minHeight: "auto",
        boxShadow: "none", fontFamily: "'Times New Roman', Times, serif"
      });
    }
    if (txt === "Reset") {
      Object.assign(b.style, {
        background: "linear-gradient(135deg,#e8a030,#d4880f)",
        color: "#fff", border: "none", borderRadius: "8px",
        fontSize: "15px", fontWeight: "700", padding: "12px",
        width: "100%", minHeight: "auto", boxShadow: "none",
        fontFamily: "'Times New Roman', Times, serif"
      });
    }
  });
}
setTimeout(applyStyles, 300);
setTimeout(applyStyles, 800);
setTimeout(applyStyles, 1500);
setTimeout(applyStyles, 3000);
P._htcObs = new MutationObserver(() => { setTimeout(applyStyles, 50); });
P._htcObs.observe(P.body, {childList:true, subtree:true});
</script>
""", height=0)

# ────────────────── HERO ──────────────────
st.markdown(f"""<div class="hero" style="padding-top:0px !important; justify-content:flex-start !important;">
<h1 style="margin-top:0px !important;">Organic Biomass HTC Multi-Product Forecaster</h1>
<p>This system uses a machine learning GBDT model optimized via Optuna to predict the yields of hydrochar, aqueous total nitrogen (TN), and carbon dot fluorescence quantum yield (QY) from organic solid waste hydrothermal carbonization (HTC).<br>Please enter the following feature parameters:</p>
</div>""", unsafe_allow_html=True)

# ────────────────── TARGET SELECTION ──────────────────
_cur = st.session_state.target
tc1, tc2, tc3 = st.columns(3)
with tc1:
    if st.button("Hydrochar Yield", use_container_width=True, key="btn_hc",
                 type="primary" if _cur=="Hydrochar Yield" else "secondary"):
        st.session_state.target = "Hydrochar Yield"; st.session_state.result = None; st.rerun()
with tc2:
    if st.button("Aqueous phase TN", use_container_width=True, key="btn_ap",
                 type="primary" if _cur=="Aqueous phase TN" else "secondary"):
        st.session_state.target = "Aqueous phase TN"; st.session_state.result = None; st.rerun()
with tc3:
    if st.button("QY of carbon dots", use_container_width=True, key="btn_cd",
                 type="primary" if _cur=="QY of carbon dots" else "secondary"):
        st.session_state.target = "QY of carbon dots"; st.session_state.result = None; st.rerun()

# ────────────────── INPUT AREA ──────────────────
# 根据当前目标加载对应训练数据的 Type 列表
_type_list, _type_map, _feat_cols = _load_type_info(st.session_state.target)
if not _type_list:
    _type_list = ["Food waste","Sewage sludge","Livestock manure",
                  "Crop straw","Woody biomass","Algae","Other"]
    _type_map  = {c: i+1 for i,c in enumerate(_type_list)}

col_L, col_M, col_R = st.columns(3)

# ===== LEFT COLUMN =====
with col_L:
    # -- Categorical Variable --
    st.markdown('<div class="fs"><span class="lg">Categorical Variable</span>', unsafe_allow_html=True)
    biomass_type = st.selectbox("Type", _type_list, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Reaction Condition Parameters --
    st.markdown('<div class="fs org"><span class="lg org">Reaction Condition Parameters</span>', unsafe_allow_html=True)
    temp  = st.number_input("Temperature (°C)", min_value=100.0, max_value=400.0, value=220.0, step=5.0, format="%.1f")
    time_ = st.number_input("Time (min)",       min_value=1.0,   max_value=1440.0, value=60.0,  step=5.0, format="%.1f")
    ratio = st.number_input("Solid-liquid ratio",min_value=0.01, max_value=1.0,    value=0.10,  step=0.01, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== MIDDLE COLUMN =====
with col_M:
    st.markdown('<div class="fs org"><span class="lg org">Elemental Analysis</span>', unsafe_allow_html=True)
    el_C = st.number_input("C (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1, format="%.2f")
    el_H = st.number_input("H (%)", min_value=0.0, max_value=100.0, value=6.0,  step=0.1, format="%.2f")
    el_O = st.number_input("O (%)", min_value=0.0, max_value=100.0, value=40.0, step=0.1, format="%.2f")
    el_N = st.number_input("N (%)", min_value=0.0, max_value=100.0, value=2.0,  step=0.1, format="%.2f")
    el_S = st.number_input("S (%)", min_value=0.0, max_value=100.0, value=0.5,  step=0.1, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== RIGHT COLUMN =====
with col_R:
    # -- Proximate Analysis --
    st.markdown('<div class="fs dk"><span class="lg dk">Proximate Analysis</span>', unsafe_allow_html=True)
    pr_M   = st.number_input("Moisture (%)",         min_value=0.0, max_value=100.0, value=8.0,  step=0.1, format="%.2f")
    pr_Ash = st.number_input("Ash (%)",              min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f")
    pr_VM  = st.number_input("Volatile matter (%)",  min_value=0.0, max_value=100.0, value=65.0, step=0.1, format="%.2f")
    pr_FC  = st.number_input("Fixed carbon (%)",     min_value=0.0, max_value=100.0, value=17.0, step=0.1, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)

    # -- Biochemical Composition --
    st.markdown('<div class="fs dk"><span class="lg dk">Biochemical Composition Analysis</span>', unsafe_allow_html=True)
    bc_CL = st.number_input("Cellulose (%)",      min_value=0.0, max_value=100.0, value=20.0, step=0.1, format="%.2f")
    bc_HC = st.number_input("Hemicellulose (%)",   min_value=0.0, max_value=100.0, value=15.0, step=0.1, format="%.2f")
    bc_LG = st.number_input("Lignin (%)",          min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f")
    bc_LP = st.number_input("Lipid (%)",           min_value=0.0, max_value=100.0, value=5.0,  step=0.1, format="%.2f")
    bc_PR = st.number_input("Protein (%)",         min_value=0.0, max_value=100.0, value=8.0,  step=0.1, format="%.2f")
    st.markdown('</div>', unsafe_allow_html=True)


# ────────────────── PREDICTION SECTION ──────────────────
st.markdown('<div class="pred-box"><span class="pred-lg">Prediction</span>', unsafe_allow_html=True)

# result display
if st.session_state.result is not None:
    st.markdown(f'<div class="pred-val">{st.session_state.target}: {st.session_state.result:.4f}</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="pred-val" style="color:#888;">Awaiting prediction…</div>',
                unsafe_allow_html=True)

# buttons
btn_L, btn_R = st.columns(2)
with btn_L:
    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
    run_clicked = st.button("Run Prediction", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with btn_R:
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    reset_clicked = st.button("Reset", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ────────────────── MODEL LOGIC ──────────────────
if reset_clicked:
    st.session_state.result = None
    st.rerun()

if run_clicked:
    cur_target = st.session_state.target
    xlsx_name, model_file, _ = TARGET_CFG[cur_target]

    # ① Type 验证
    if biomass_type not in _type_map:
        st.error(f"⚠️ The selected Type「{biomass_type}」was not present in the "
                 f"training data for **{cur_target}**. Prediction is not possible.\n\n"
                 f"Valid Types: {', '.join(_type_map.keys())}")
    else:
        # ② 构建特征行（列顺序与训练数据完全一致）
        raw_vals = {
            "Type": _type_map[biomass_type],   # 数值编码，从 1 开始
            "Temperature": temp, "Time": time_, "Solid-liquid ratio": ratio,
            "C": el_C, "H": el_H, "O": el_O, "N": el_N, "S": el_S,
            "Moisture": pr_M, "Ash": pr_Ash, "VM": pr_VM, "FC": pr_FC,
            "Cellulose": bc_CL, "Hemicellulose": bc_HC, "Lignin": bc_LG,
            "Lipid": bc_LP, "Protein": bc_PR,
        }
        # 按训练数据列顺序排列；如果列名不完全匹配则退回固定顺序
        if _feat_cols:
            try:
                ordered = [raw_vals[c] for c in _feat_cols]
            except KeyError:
                # 列名可能不同——用 positional fallback
                ordered = list(raw_vals.values())
        else:
            ordered = list(raw_vals.values())

        features = np.array([ordered])

        # ③ 加载模型 & 预测
        model_path = _APP_DIR / model_file
        if not model_path.exists():
            _ensure_file(model_file)   # 尝试从 GitHub 下载

        if model_path.exists():
            try:
                model = joblib.load(str(model_path))
                pred = float(model.predict(features)[0])
                st.session_state.result = pred
                st.rerun()
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.warning(f"Model file not found: {model_file}. "
                       f"Please place it in {_APP_DIR}")
