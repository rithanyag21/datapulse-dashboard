import ast, contextlib, joblib, pandas as pd, streamlit as st
from pathlib import Path
ROOT = Path(__file__).parent
P = dict(pur='#7C5CFF', ora='#FF9F43', pos='#2ED573', neu='#A4B0BE', neg='#FF4757')
SC = {'Positive': P['pos'], 'Neutral': P['neu'], 'Negative': P['neg']}
CSS = """<style>
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#2a1f5c 0%,#0E0B1F 55%)}
.hero{padding:22px 28px;border-radius:18px;background:linear-gradient(120deg,#5b3df5,#8b5cf6 55%,#ff9f43);margin-bottom:16px}
.hero h1{margin:0;color:#fff;font-size:2rem}.hero p{margin:4px 0 0;color:#fff;opacity:.92}
.kpi{background:#1A1533;border:1px solid #2f2755;border-radius:14px;padding:14px 16px}
.kpi b{font-size:1.65rem;color:#fff}.kpi span{color:#a99fd6;font-size:.8rem;display:block}.kpi i{color:#ff9f43;font-style:normal;font-size:.75rem}
.card{background:#1A1533;border-left:4px solid #FF9F43;border-radius:10px;padding:12px 16px;margin:8px 0;color:#e6e1ff}
.warn{border-left-color:#FF4757}
.flow{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
.chip{background:#231c47;border:1px solid #7C5CFF;border-radius:12px;padding:8px 12px;color:#fff;font-size:.85rem}
.arr{color:#FF9F43;font-size:1.3rem}
</style>"""

def setup(title, sub):
    st.set_page_config(page_title='DATA PULSE | Social Engine', page_icon='🌀', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### 🌀 DATA PULSE\n**RITHANYA G**  \nSRM Institute of Science & Technology  \nAaruush '26 · Data Vortex · Round 4")
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def style(f, h=380):
    f.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=h, margin=dict(l=10, r=10, t=45, b=10), colorway=[P['pur'], P['ora'], P['pos'], P['neg'], P['neu']])
    st.plotly_chart(f, use_container_width=True)

def kpis(items):
    for c, (l, v, s) in zip(st.columns(len(items)), items):
        c.markdown(f'<div class="kpi"><span>{l}</span><b>{v}</b><i>{s}</i></div>', unsafe_allow_html=True)

def note(t, warn=False):
    st.markdown(f'<div class="card{" warn" if warn else ""}">{t}</div>', unsafe_allow_html=True)

def flow(steps):
    st.markdown('<div class="flow">' + '<span class="arr">➜</span>'.join(f'<span class="chip">{s}</span>' for s in steps) + '</div>', unsafe_allow_html=True)

@st.cache_data
def posts(): return pd.read_csv(ROOT / 'data/posts_v2.csv', parse_dates=['timestamp'])
@st.cache_data
def users(): return pd.read_csv(ROOT / 'data/users.csv')
@st.cache_data
def r3():
    d = pd.read_csv(ROOT / 'data/r3_scored.csv'); d['themes'] = d.themes.map(ast.literal_eval); return d
@st.cache_resource
def model(): return joblib.load(ROOT / 'models/tfidf_vectorizer.pkl'), joblib.load(ROOT / 'models/final_nlp_model.pkl')
def predict(texts):
    v, m = model(); return m.classes_, m.predict_proba(v.transform(list(texts)))

@contextlib.contextmanager
def guard(msg='This chart is unavailable for the current filter selection.'):
    try: yield
    except Exception: st.caption(msg)
