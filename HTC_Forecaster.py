# -*- coding: utf-8 -*-
"""Organic Biomass HTC Multi-Product Forecaster"""
import streamlit as st, numpy as np, os, joblib, base64, urllib.request, pathlib, pandas as pd

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
  font-size:28px!important;font-weight:700!important;color:#fff!important;
  border:none!important;border-radius:10px!important;
  padding:6px 10px!important;min-height:auto!important;
  line-height:1.2!important;
  transition:all .25s ease!important;
}}
/* secondary 按钮内的 p 标签也要改字号 */
button[data-testid="stBaseButton-secondary"] p,
button[data-testid="stBaseButton-primary"] p{{
  font-size:28px!important;font-family:'Times New Roman',Times,serif!important;
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

/* ============ FIELDSET — 全部由JS处理，CSS只保留辅助样式 ============ */
/* 强制清除所有 BorderWrapper 的 Streamlit 默认边框 */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border:none!important;box-shadow:none!important;
  background:transparent!important;overflow:visible!important;
  padding:0!important;margin:0!important;
}}

/* --- label / unit cells --- */
.lab-cell{{font-family:'Times New Roman',Times,serif;font-weight:700;
           font-size:28px;display:flex!important;align-items:center!important;
           min-height:44px!important;padding-left:4px}}
.unit-cell{{font-family:'Times New Roman',Times,serif;font-size:28px;
            display:flex!important;align-items:center!important;
            min-height:44px!important;padding-left:6px}}

/* 每行垂直居中 — 用后代选择符，不用 > */
[data-testid="stHorizontalBlock"]{{
  display:flex!important;align-items:center!important;
}}
[data-testid="stColumn"]{{
  display:flex!important;flex-direction:column!important;
  justify-content:center!important;align-self:center!important;
  padding-top:2px!important;padding-bottom:2px!important;
}}
/* stNumberInput 和 stSelectbox 容器也垂直居中 */
[data-testid="stNumberInput"],[data-testid="stSelectbox"]{{
  display:flex!important;flex-direction:column!important;justify-content:center!important;
}}

/* ★ 只清除内层，不清除 [data-baseweb="input"]（由JS加彩色边框） */
[data-baseweb="base-input"],
[data-baseweb="input-container"],
[data-baseweb="form-control"]{{
  border:none!important;box-shadow:none!important;background:transparent!important;
}}
[data-baseweb="input-adjoin"]{{display:none!important;}}
[data-baseweb="select"] div,[data-baseweb="select"]>div{{
  border:none!important;box-shadow:none!important;
}}

/* 隐藏 number_input 的 +/- 按钮 */
[data-testid="stNumberInput"] button{{display:none!important}}
[data-testid="stNumberInput"]>div{{
  border-radius:8px!important;box-sizing:border-box!important;
  margin-left:0!important;padding-left:0!important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"]>div,
[data-testid="stNumberInput"]>div{{
  display:flex!important;align-items:center!important;min-height:44px!important;
}}
[data-testid="stNumberInput"] input{{
  font-family:'Times New Roman',Times,serif!important;
  font-size:22px!important;padding:8px 12px!important;
  background:transparent!important;color:#222!important;
  width:100%!important;flex:1!important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] > div{{
  font-family:'Times New Roman',Times,serif!important;
  font-size:22px!important;min-height:44px!important;
  background:#fff!important;color:#222!important;
  width:100%!important;box-sizing:border-box!important;
}}
label,[data-testid="stWidgetLabel"]{{color:#333!important}}

/* ── Target Selection 框和 legend ── */
/* 匹配所有可能的容器标记方式 */
div[data-testid="stHorizontalBlock"][data-ts-styled],
div[data-testid="stHorizontalBlock"].ts-box,
div[data-testid="stHorizontalBlock"].ts-container{{
  border:2px solid #1b2a4a!important;border-radius:12px!important;
  padding:30px 0px 14px!important;position:relative!important;
  margin-bottom:14px!important;overflow:visible!important;
  display:flex!important;justify-content:center!important;
  align-items:center!important;gap:20px!important;
}}
/* 容器内每列等宽居中 */
div[data-testid="stHorizontalBlock"][data-ts-styled] > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"].ts-box > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"].ts-container > div[data-testid="stColumn"]{{
  display:flex!important;justify-content:center!important;
  flex:1 1 0!important;
}}
/* 匹配旧JS创建的无class span + 新JS创建的span */
div[data-testid="stHorizontalBlock"][data-ts-styled] > span,
div[data-testid="stHorizontalBlock"].ts-box > span,
div[data-testid="stHorizontalBlock"].ts-container > span{{
  position:absolute!important;top:-18px!important;left:50%!important;
  transform:translateX(-50%)!important;background:var(--background-color,#fff)!important;
  padding:0 14px!important;font-family:'Times New Roman',Times,serif!important;
  font-weight:700!important;font-size:33px!important;color:#1b2a4a!important;
  white-space:nowrap!important;z-index:10!important;line-height:1.1!important;
}}

/* run / reset — 由 JS 定位 */
.pred-outer{{
  position:relative;border:3px solid #1b2a4a;border-radius:14px;
  padding:20px 20px 16px;margin-top:18px;
}}
.pred-legend{{
  position:absolute;top:-20px;left:50%;transform:translateX(-50%);
  background:#fff;padding:0 14px;
  font-family:'Times New Roman',Times,serif;font-weight:700;
  font-size:30px;color:#1b2a4a;white-space:nowrap;z-index:10;
}}
.pred-left{{
  background:#1b2a4a;border-radius:10px;
  padding:12px 20px 12px 250px;display:flex;align-items:center;gap:12px;
  box-sizing:border-box;align-self:stretch;
  border:none!important;outline:none!important;
}}
.pred-label{{
  font-family:'Times New Roman',Times,serif;font-weight:700;
  font-size:32px;color:#fff;white-space:nowrap;
}}
.pred-value{{
  font-family:'Times New Roman',Times,serif;font-weight:700;
  font-size:32px;color:#fff;
}}
/* Prediction外框标题 */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.pred-marker){{
  position:relative!important;overflow:visible!important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.pred-marker)::before{{
  content:'Prediction';
  position:absolute;top:-18px;left:50%;transform:translateX(-50%);
  background:#fff;padding:0 14px;
  font-family:'Times New Roman',Times,serif;font-weight:700;
  font-size:30px;color:#1b2a4a;white-space:nowrap;z-index:10;line-height:1.5;
}}
.pred-outer button:first-of-type{{
  background:#4995AD!important;color:#fff!important;
  font-family:'Times New Roman',Times,serif!important;
  font-size:20px!important;font-weight:700!important;
  border:none!important;border-radius:8px!important;
}}
.pred-outer button:last-of-type{{
  background:linear-gradient(135deg,#e8a030,#d4880f)!important;color:#fff!important;
  font-family:'Times New Roman',Times,serif!important;
  font-size:20px!important;font-weight:700!important;
  border:none!important;border-radius:8px!important;
}}

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
          tgtBlock.classList.add('ts-box');
          /* 如果容器内还没有 legend span，就创建一个 */
          let hasLegend = false;
          tgtBlock.querySelectorAll('span').forEach(s => {
            if ((s.textContent || '').trim() === 'Target Selection') hasLegend = true;
          });
          if (!hasLegend) {
            const legend = P.createElement('span');
            legend.textContent = 'Target Selection';
            tgtBlock.insertBefore(legend, tgtBlock.firstChild);
          }
        }
      }
      /* ── 按钮样式 ── */
      const kind = b.getAttribute('kind');
      Object.assign(b.style, {
        fontFamily: "'Times New Roman', Times, serif",
        fontSize: "28px", fontWeight: "700", color: "#fff",
        border: "none", borderRadius: "10px",
        padding: "6px 10px", minHeight: "auto", lineHeight: "1.2",
        transition: "all .25s ease", width: "75%", margin: "0 auto", display: "block"
      });
      const p = b.querySelector('p');
      if (p) { p.style.fontSize = "28px"; p.style.fontFamily = "'Times New Roman', Times, serif"; p.style.fontWeight = "700"; p.style.margin = "0"; p.style.lineHeight = "1.2"; }
      if (kind === "primary") {
        b.style.background = "#c0392b";
        b.style.boxShadow = "0 4px 16px rgba(192,57,43,.35)";
      } else {
        b.style.background = "#1b2a4a";
        b.style.boxShadow = "none";
      }
    }
    if (txt === "Run Prediction") {
      b.style.setProperty('background','linear-gradient(135deg,#2e8b9a,#5bb8c4)','important');
      b.style.setProperty('color','#fff','important');
      b.style.setProperty('border','none','important');
      b.style.setProperty('border-radius','8px','important');
      b.style.setProperty('font-size','22px','important');
      b.style.setProperty('font-weight','700','important');
      b.style.setProperty('font-family',"'Times New Roman',Times,serif",'important');
    }
    if (txt === "Reset Inputs") {
      b.style.setProperty('background','linear-gradient(135deg,#e8a030,#f0c060)','important');
      b.style.setProperty('color','#fff','important');
      b.style.setProperty('border','none','important');
      b.style.setProperty('border-radius','8px','important');
      b.style.setProperty('font-size','22px','important');
      b.style.setProperty('font-weight','700','important');
      b.style.setProperty('font-family',"'Times New Roman',Times,serif",'important');
      b.style.setProperty('margin-top','12px','important');
    }
  });

  /* ── fieldset：从 lab-cell 向上找容器 ── */
  const FS_MAP = [
    {marker:'Type',  label:'Categorical Variable',            color:'#c0392b'},
    {marker:'T',     label:'Reaction Condition Parameters',   color:'#1b2a4a'},
    {marker:'C',     label:'Elemental Analysis',              color:'#F08F3E'},
    {marker:'Protein', label:'Proximate Analysis',            color:'#d4880f'},
    {marker:'FC',    label:'Biochemical Composition Analysis',color:'#4995AD'},
  ];
  const labCells = P.querySelectorAll('.lab-cell');
  console.log('[HTC-FS] lab-cells:', labCells.length);
  /* 调试：打印第一个 lab-cell 的所有祖先 testid */
  if (labCells.length > 0) {
    let dbg = [], c = labCells[0];
    for (let i=0; i<20; i++) { c=c.parentElement; if(!c||c===P.body) break; dbg.push((c.getAttribute('data-testid')||'?')+'['+c.tagName+']'); }
    console.log('[HTC-FS] ancestors:', dbg.join(' > '));
  }
  /* ── 注入 <style> 到 head 末尾 ── */
  let _ks = P.getElementById('htc-kill-style');
  if (!_ks) { _ks=P.createElement('style'); _ks.id='htc-kill-style'; P.head.appendChild(_ks); }
  _ks.textContent = `
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.lab-cell){border:none!important;box-shadow:none!important;padding:0!important;margin:0!important;background:transparent!important;overflow:visible!important;}
    [data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="input-container"]{border:none!important;box-shadow:none!important;background:transparent!important;}
    [data-baseweb="input-adjoin"]{display:none!important;}
    [data-baseweb="select"]>div{border:none!important;box-shadow:none!important;}
    [data-testid="stWidgetLabel"]{display:none!important;}
    [data-testid="stNumberInput"],[data-testid="stSelectbox"]{margin:0!important;}
    [data-testid="stHorizontalBlock"]{display:flex!important;align-items:center!important;margin:2px 0!important;}
    [data-testid="stColumn"]{align-self:center!important;}
    .lab-cell,.unit-cell{display:flex!important;align-items:center!important;min-height:44px!important;}
    [data-testid="stVerticalBlock"]{gap:0!important;}
    [data-testid="stVerticalBlock"]>.element-container{margin-top:0!important;margin-bottom:0!important;padding-top:0!important;padding-bottom:0!important;}
    .pred-left hr,.pred-marker hr{display:none!important;}
    .pred-left .element-container,.pred-marker .element-container{border:none!important;box-shadow:none!important;}
    .pred-marker *{border-top:none!important;border-bottom:none!important;box-shadow:none!important;}
    [data-testid="stMarkdownContainer"]{border:none!important;border-bottom:none!important;}
    .pred-marker [data-testid="stMarkdownContainer"]{border:none!important;border-bottom:none!important;box-shadow:none!important;}
    .pred-marker p,.pred-marker span,.pred-marker div{border:none!important;border-bottom:none!important;}
    .pred-left,.pred-left *,.pred-marker,.pred-marker *{border-bottom:none!important;border-top:none!important;}
  `;

  const seen = new Set();
  labCells.forEach(cell => {
    const txt = (cell.textContent||'').trim();
    const m = FS_MAP.find(x => x.marker === txt);
    if (!m) return;
    /* 向上找：先找到任意 stVerticalBlock，再继续往上找包含它的 stVerticalBlock（即外层容器） */
    let inner = null, wrapper = null, cur = cell;
    for (let i=0; i<20; i++) {
      cur = cur.parentElement;
      if (!cur || cur === P.body) break;
      const tid = cur.getAttribute('data-testid')||'';
      if (tid === 'stVerticalBlock') {
        if (!inner) { inner = cur; }
        else { wrapper = cur; break; }
      }
    }
    if (!wrapper) wrapper = inner;
    if (!wrapper || seen.has(wrapper)) return;
    seen.add(wrapper);

    /* 强制清除 BorderWrapper 的 Streamlit 原始边框（用 setProperty 才能覆盖） */
    let bw = wrapper.parentElement;
    for (let i=0; i<8; i++) {
      if (!bw || bw===P.body) break;
      const tid = bw.getAttribute('data-testid')||'';
      if (tid.includes('BorderWrapper') || tid.includes('stVerticalBlock')) {
        ['border','box-shadow','padding','background'].forEach(p=>
          bw.style.setProperty(p, p==='background'?'transparent':'none', 'important')
        );
        bw.style.setProperty('overflow','visible','important');
      }
      bw = bw.parentElement;
    }

    /* 内层容器：彩色边框，中列padding-top 48px，其余28px */
    const isMiddle = (m.marker === 'C');
    const isSecondRight = (m.marker === 'FC');
    const isT = (m.marker === 'T');
    const ptop = isMiddle ? '82px' : (isT ? '52px' : '28px');
    const isFirst = (m.marker === 'Type' || m.marker === 'C' || m.marker === 'Protein');
    const mtop = isSecondRight ? '15px' : (isFirst ? '10px' : '38px');
    const pbot = isMiddle ? '80px' : '12px';
    ['border','border-radius','position','overflow','padding',
     'margin-top','margin-bottom','background','box-sizing'].forEach((p,i)=>{
      const vals=['3px solid '+m.color,'14px','relative','visible',
                  ptop+' 12px '+pbot, mtop,'10px','transparent','border-box'];
      wrapper.style.setProperty(p, vals[i], 'important');
    });

    /* 浮动标题：背景#fff遮断框线 */
    let span = wrapper.querySelector('.fs-span');
    if (!span) { span=P.createElement('span'); span.className='fs-span'; wrapper.insertBefore(span,wrapper.firstChild); }
    span.textContent = m.label;
    [['position','absolute'],['top','-18px'],['left','50%'],
     ['transform','translateX(-50%)'],['background','#ffffff'],
     ['color',m.color],['padding','0 10px'],['font-size','26px'],
     ['font-weight','700'],['font-family',"'Times New Roman',Times,serif"],
     ['white-space','nowrap'],['z-index','10'],['line-height','1.5'],
     ['border-radius','0']].forEach(([p,v])=>span.style.setProperty(p,v,'important'));

  /* ── 全局扫描所有 stNumberInput，找同行 lab-cell 确定颜色，加彩色边框 ── */
  const COLOR_MAP = {
    'Type':'#c0392b','T':'#1b2a4a','RT':'#1b2a4a','SLR':'#1b2a4a','Cycles':'#1b2a4a',
    'C':'#F08F3E','H':'#F08F3E','O':'#F08F3E','N':'#F08F3E','S':'#F08F3E',
    'M':'#2c3e50','Ash':'#2c3e50','VM':'#2c3e50','FC':'#2c3e50',
    'Protein':'#d4880f','Lipid':'#d4880f','CHO':'#d4880f',
    'FC':'#4995AD','VM':'#4995AD','Ash':'#4995AD',
  };
  /* 清除所有 input 内层边框 */
  P.querySelectorAll('[data-baseweb="input"],[data-baseweb="base-input"]').forEach(el=>{
    el.style.setProperty('border','none','important');
    el.style.setProperty('box-shadow','none','important');
    el.style.setProperty('background','transparent','important');
  });
  P.querySelectorAll('[data-baseweb="input-adjoin"]').forEach(el=>{
    el.style.setProperty('display','none','important');
  });
  /* 对每个 stNumberInput，找同行 lab-cell 文字，加对应颜色边框 */
  P.querySelectorAll('[data-testid="stNumberInput"]').forEach(ni=>{
    /* 向上找 stHorizontalBlock */
    let row=ni;
    for(let i=0;i<10;i++){
      row=row.parentElement;
      if(!row||row===P.body) break;
      if((row.getAttribute('data-testid')||'')==='stHorizontalBlock') break;
    }
    if(!row) return;
    /* 在同行找 lab-cell 文字 */
    const lc=row.querySelector('.lab-cell');
    const txt=lc?(lc.textContent||'').trim():'';
    const color=COLOR_MAP[txt]||'#888';
    /* lab-cell 和 unit-cell 字体颜色 */
    if(lc) lc.style.setProperty('color',color,'important');
    const uc=row.querySelector('.unit-cell');
    if(uc) uc.style.setProperty('color',color,'important');
    /* stNumberInput 直接子div加彩色边框 */
    ni.querySelectorAll('[data-baseweb="input-adjoin"]').forEach(el=>{
      el.style.setProperty('display','none','important');
    });
    const firstDiv = ni.querySelector(':scope > div');
    if(firstDiv){
      const selDiv = P.querySelector('[data-testid="stSelectbox"] [data-baseweb="select"]>div');
      const refW = selDiv ? selDiv.getBoundingClientRect().width : 0;
      if(refW>0) firstDiv.style.setProperty('width', refW+'px','important');
      /* 基准：Food waste框左边缘的绝对x坐标 */
      const refLeft = selDiv ? selDiv.getBoundingClientRect().left : 0;
      /* 当前number_input框左边缘的绝对x坐标 */
      const curLeft = firstDiv.getBoundingClientRect().left;
      /* 找到包含firstDiv的stColumn，调整padding-left对齐 */
      if(refLeft>0 && curLeft !== refLeft){
        let col=ni;
        for(let i=0;i<8;i++){col=col.parentElement;if(!col||col===P.body)break;if((col.getAttribute('data-testid')||'')==='stColumn')break;}
        if(col&&(col.getAttribute('data-testid')||'')==='stColumn'){
          const curPL=parseFloat(getComputedStyle(col).paddingLeft)||0;
          const diff=curLeft-refLeft;
          col.style.setProperty('padding-left',Math.max(0,curPL-diff)+'px','important');
        }
      }
      firstDiv.style.setProperty('border','2px solid '+color,'important');
      firstDiv.style.setProperty('border-radius','8px','important');
      firstDiv.style.setProperty('box-shadow','none','important');
      firstDiv.style.setProperty('background','#fff','important');
    }
    ni.querySelectorAll('[data-baseweb="input"],[data-baseweb="base-input"]').forEach(el=>{
      el.style.setProperty('border','none','important');
      el.style.setProperty('box-shadow','none','important');
      el.style.setProperty('background','transparent','important');
    });
  });
  /* selectbox: 从 lab-cell 找同行 select>div 加彩色边框 */
  P.querySelectorAll('.lab-cell').forEach(cell=>{
    const txt=(cell.textContent||'').trim();
    const color=COLOR_MAP[txt];
    if(!color) return;
    cell.style.setProperty('color',color,'important');
    let row=cell;
    for(let i=0;i<10;i++){
      row=row.parentElement;
      if(!row||row===P.body) break;
      if((row.getAttribute('data-testid')||'')==='stHorizontalBlock') break;
    }
    if(!row) return;
    const uc=row.querySelector('.unit-cell');
    if(uc) uc.style.setProperty('color',color,'important');
    row.querySelectorAll('[data-baseweb="select"]>div').forEach(el=>{
      el.style.setProperty('border','2px solid '+color,'important');
      el.style.setProperty('border-radius','8px','important');
      el.style.setProperty('box-shadow','none','important');
    });
  });

    /* 每行垂直居中，不改宽度 */
    wrapper.querySelectorAll('[data-testid="stHorizontalBlock"]').forEach(row=>{
      row.style.setProperty('display','flex','important');
      row.style.setProperty('align-items','center','important');
      row.querySelectorAll('[data-testid="stColumn"]').forEach(col=>{
        col.style.setProperty('align-self','center','important');
      });
    });
    console.log('[HTC-FS] styled:', m.label);
  });
}
function equalizeColumns() {
  let mainHB = null;
  P.querySelectorAll('[data-testid="stHorizontalBlock"]').forEach(hb => {
    const texts = Array.from(hb.querySelectorAll('.lab-cell')).map(c=>(c.textContent||'').trim());
    if (texts.includes('Type') && texts.includes('C') && texts.includes('FC')) mainHB = hb;
  });
  if (!mainHB) return;
  mainHB.style.setProperty('align-items','flex-start','important');

  /* 收集三列的 stVerticalBlock */
  const vbs = [];
  Array.from(mainHB.children).forEach(col => {
    if ((col.getAttribute('data-testid')||'') !== 'stColumn') return;
    const vb = col.querySelector('[data-testid="stVerticalBlock"]');
    if (!vb) return;
    /* 清除旧 spacer */
    const old = vb.querySelector('.htc-eq-spacer');
    if (old) old.remove();
    vbs.push(vb);
  });
  if (vbs.length < 2) return;

  /* 实测各列内容高度，找最大值 */
  const heights = vbs.map(vb => vb.getBoundingClientRect().height);
  const maxH = Math.max(...heights);

  /* 给高度不足的列底部补 spacer */
  vbs.forEach((vb, i) => {
    const diff = maxH - heights[i];
    if (diff > 2) {
      const sp = P.createElement('div');
      sp.className = 'htc-eq-spacer';
      sp.style.cssText = 'height:' + diff + 'px;flex-shrink:0;';
      vb.appendChild(sp);
    }
  });
}
function resetAndApply() {
  P.querySelectorAll('[data-fs-styled]').forEach(el => delete el.dataset.fsStyled);
  applyStyles();
  stylePredTitle();
}
function stylePredTitle() {
  const pm = P.querySelector('.pred-marker');
  if (!pm) return;
  /* 找包含pred-marker的stHorizontalBlock，再往上一层stVerticalBlock */
  let bw = null, cur = pm;
  for (let i=0; i<20; i++) {
    cur = cur.parentElement;
    if (!cur || cur===P.body) break;
    if ((cur.getAttribute('data-testid')||'')==='stHorizontalBlock') {
      /* 再往上找stVerticalBlock */
      let p2 = cur.parentElement;
      for (let j=0; j<5; j++) {
        if (!p2||p2===P.body) break;
        if ((p2.getAttribute('data-testid')||'')==='stVerticalBlock') { bw=p2; break; }
        p2=p2.parentElement;
      }
      /* 对齐左侧色块与右侧按钮顶部 */
      const cols = Array.from(cur.querySelectorAll(':scope > [data-testid="stColumn"]'));
      if (cols.length >= 2) {
        const leftTop = cols[0].getBoundingClientRect().top;
        const rightTop = cols[1].getBoundingClientRect().top;
        const rightH = cols[1].getBoundingClientRect().height;
        const diff = rightTop - leftTop;
        if (rightH > 10) pm.style.setProperty('height', rightH+'px', 'important');
        if (Math.abs(diff) > 1) pm.style.setProperty('margin-top', diff+'px', 'important');
      }
      break;
    }
  }
  if (!bw) return;
  /* 清除pred-marker所有祖先的border-bottom直到bw */
  let up = pm.parentElement;
  while (up && up !== bw) {
    up.style.setProperty('border-bottom','none','important');
    up.style.setProperty('border-top','none','important');
    up.style.setProperty('box-shadow','none','important');
    up = up.parentElement;
  }
  ['border','border-radius','position','overflow','padding','margin-top']
    .forEach((p,i)=>bw.style.setProperty(p,
      ['3px solid #1b2a4a','14px','relative','visible','28px 16px 16px','18px'][i],'important'));
  if (bw.querySelector('.pred-legend-span')) return;
  const leg=P.createElement('span');
  leg.className='pred-legend-span';
  leg.textContent='Prediction';
  [['position','absolute'],['top','-18px'],['left','50%'],['transform','translateX(-50%)'],
   ['background','#fff'],['color','#1b2a4a'],['padding','0 14px'],['font-size','30px'],
   ['font-weight','700'],['font-family',"'Times New Roman',Times,serif"],
   ['white-space','nowrap'],['z-index','10'],['line-height','1.5']]
    .forEach(([p,v])=>leg.style.setProperty(p,v,'important'));
  bw.insertBefore(leg,bw.firstChild);
}
setTimeout(applyStyles, 200);
setTimeout(applyStyles, 600);
setTimeout(applyStyles, 1200);
setTimeout(applyStyles, 2500);
setTimeout(stylePredTitle, 800);
setTimeout(stylePredTitle, 1600);
setTimeout(equalizeColumns, 400);
setTimeout(equalizeColumns, 800);
setTimeout(equalizeColumns, 1600);
setTimeout(equalizeColumns, 3000);
let _iv = setInterval(resetAndApply, 2000);
setTimeout(() => clearInterval(_iv), 60000);
P._htcObs = new MutationObserver(() => { setTimeout(applyStyles, 100); });
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

st.markdown('<div style="margin-top:1px"></div>', unsafe_allow_html=True)
col_L, _g1, col_M, _g2, col_R = st.columns([0.25, 0.05, 0.25, 0.05, 0.25])

# ===== LEFT COLUMN =====
with col_L:
    with st.container(border=True):
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c:
            st.markdown('<div class="lab-cell" style="margin-top:-16px">Type</div>', unsafe_allow_html=True)
        with in_c:
            biomass_type = st.selectbox("Type", _type_list, label_visibility="collapsed")
        with u_c:
            st.markdown('<div class="unit-cell" style="margin-top:-16px"></div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">T</div>', unsafe_allow_html=True)
        with in_c:  temp = st.number_input("T", min_value=100.0, max_value=400.0, value=220.0, step=5.0, format="%.1f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">°C</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">RT</div>', unsafe_allow_html=True)
        with in_c:  time_ = st.number_input("RT", min_value=1.0, max_value=1440.0, value=60.0, step=5.0, format="%.1f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">h</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">SLR</div>', unsafe_allow_html=True)
        with in_c:  ratio = st.number_input("SLR", min_value=0.01, max_value=1.0, value=0.10, step=0.01, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">Cycles</div>', unsafe_allow_html=True)
        with in_c:  cycles = st.number_input("Cycles", min_value=1, max_value=100, value=1, step=1, format="%d", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">times</div>', unsafe_allow_html=True)

# ===== MIDDLE COLUMN =====
with col_M:
    with st.container(border=True):
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">C</div>', unsafe_allow_html=True)
        with in_c:  el_C = st.number_input("C", min_value=0.0, max_value=100.0, value=45.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">H</div>', unsafe_allow_html=True)
        with in_c:  el_H = st.number_input("H", min_value=0.0, max_value=100.0, value=6.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">O</div>', unsafe_allow_html=True)
        with in_c:  el_O = st.number_input("O", min_value=0.0, max_value=100.0, value=40.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">N</div>', unsafe_allow_html=True)
        with in_c:  el_N = st.number_input("N", min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">S</div>', unsafe_allow_html=True)
        with in_c:  el_S = st.number_input("S", min_value=0.0, max_value=100.0, value=0.5, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

# ===== RIGHT COLUMN =====
with col_R:
    with st.container(border=True):
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">Protein</div>', unsafe_allow_html=True)
        with in_c:  pr_Protein = st.number_input("Protein", min_value=0.0, max_value=100.0, value=8.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">Lipid</div>', unsafe_allow_html=True)
        with in_c:  pr_Lipid = st.number_input("Lipid", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">CHO</div>', unsafe_allow_html=True)
        with in_c:  pr_CHO = st.number_input("CHO", min_value=0.0, max_value=100.0, value=65.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:15px"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">FC</div>', unsafe_allow_html=True)
        with in_c:  bc_FC = st.number_input("FC", min_value=0.0, max_value=100.0, value=17.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">VM</div>', unsafe_allow_html=True)
        with in_c:  bc_VM = st.number_input("VM", min_value=0.0, max_value=100.0, value=65.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        lab_c, in_c, u_c = st.columns([1, 2, 0.6])
        with lab_c: st.markdown('<div class="lab-cell" style="margin-top:-16px">Ash</div>', unsafe_allow_html=True)
        with in_c:  bc_Ash = st.number_input("Ash_bc", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f", label_visibility="collapsed")
        with u_c:   st.markdown('<div class="unit-cell" style="margin-top:-16px;padding-left:12px">wt%</div>', unsafe_allow_html=True)


# ────────────────── PREDICTION SECTION ──────────────────
with st.container(border=False):
    pred_L, pred_R = st.columns([3, 1])
    with pred_L:
        if st.session_state.result is not None:
            val = f"{st.session_state.target}: {st.session_state.result:.4f}"
        else:
            val = ""
        st.markdown(f'''<div class="pred-left pred-marker">
          <span class="pred-label">Predicted value:</span>
          <span class="pred-value">{val}</span>
        </div>''', unsafe_allow_html=True)
    with pred_R:
        run_clicked = st.button("Run Prediction", use_container_width=True)
        st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
        reset_clicked = st.button("Reset Inputs", use_container_width=True)

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