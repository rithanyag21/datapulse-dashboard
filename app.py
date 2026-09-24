import re, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from common import *
setup('Social Engine Revival', 'Intake → SQL Analytics → NLP Semantics → Live Monitoring · one connected pipeline')
raw = pd.read_csv(ROOT / 'data/posts_corrupted.csv'); p = posts(); u = users(); l = r3()
st.subheader('Round 1 · Phase 1 — Data Intake')
kpis([('Raw rows', f'{len(raw):,}', 'corrupted intake'), ('Clean posts', f'{len(p):,}', f'{len(raw)-len(p)} duplicates removed'),
      ('Users', f'{len(u):,}', 'validated, 0 issues'), ('Likes repaired', f'{p.likes_imputed.sum():,}', 'missing or negative → median'),
      ('Live records (R3)', len(l), 'Ars + Hacker News')])
t1, t2, t3 = st.tabs(['🧹 Cleaning audit', '📊 EDA explorer', '🔗 End-to-end pipeline'])
with t1, guard("This section is temporarily unavailable."):
    rd = raw.drop_duplicates()
    fm = raw.timestamp.astype(str).map(lambda s: 'Unix epoch' if re.fullmatch(r'\d{9,11}', s) else 'ISO 8601' if 'T' in s else 'DD-MM-YYYY')
    a, b = st.columns(2)
    with a:
        miss = pd.DataFrame({'Field': ['platform', 'text_content', 'likes'], 'Corrupted': [rd.platform.isna().sum(), rd.text_content.isna().sum(), rd.likes.isna().sum()], 'Cleaned': [0, 0, 0]})
        style(px.bar(miss, x='Field', y=['Corrupted', 'Cleaned'], barmode='group', title='Missing values before vs after (after de-dup)'))
    with b:
        style(px.pie(fm.value_counts().reset_index(), names='timestamp', values='count', hole=.55, title='3 timestamp formats unified'))
    a, b = st.columns(2)
    with a:
        h = pd.concat([pd.DataFrame({'likes': rd.likes.dropna(), 'v': 'Corrupted'}), pd.DataFrame({'likes': p.likes, 'v': 'Cleaned'})])
        style(px.histogram(h, x='likes', color='v', barmode='overlay', nbins=60, opacity=.7, title='Likes: negatives removed, median spike at 2502'))
    with b:
        st.dataframe(pd.DataFrame({'Issue': ['Duplicate rows', 'Missing platform', 'Missing text', 'Missing likes', 'Negative likes', 'Mixed timestamps'],
            'Found': [len(raw) - len(rd), rd.platform.isna().sum(), rd.text_content.isna().sum(), rd.likes.isna().sum(), (rd.likes < 0).sum(), '3 formats'],
            'Treatment': ['Dropped', "'Unknown'", "'[Missing text]'", 'Median (2502)', 'Median (2502)', 'One ISO datetime']}), hide_index=True, use_container_width=True)
    note('<b>Timestamp integrity:</b> ISO-format dates were re-parsed from the raw file (day/month kept in the right order). The data spans <b>1 May 2024 – 30 Apr 2025</b>. Date-only rows stay at 00:00; nothing was fabricated. An <code>likes_imputed</code> flag keeps every repaired value auditable.')
with t2, guard("This section is temporarily unavailable."):
    c1, c2 = st.columns([2, 1])
    pf = c1.multiselect('Platforms', sorted(p.platform.unique()), default=[x for x in sorted(p.platform.unique()) if x != 'Unknown'])
    dr = c2.date_input('Date range', (p.timestamp.min().date(), p.timestamp.max().date()))
    f = p[p.platform.isin(pf)]
    if len(dr) == 2: f = f[(f.timestamp.dt.date >= dr[0]) & (f.timestamp.dt.date <= dr[1])]
    f = f.assign(eng=f.likes + f.shares + f.comments, month=f.timestamp.dt.to_period('M').astype(str))
    a, b = st.columns(2)
    with a: style(px.line(f.groupby(['month', 'platform']).size().reset_index(name='posts'), x='month', y='posts', color='platform', markers=True, title='Monthly post volume'))
    with b: style(px.bar(f.groupby('platform').eng.mean().reset_index(), x='platform', y='eng', color='platform', title='Avg engagement per post'))
    t = f[(f.timestamp.dt.hour > 0) | (f.timestamp.dt.minute > 0)]
    hm = t.groupby([t.timestamp.dt.day_name().rename('day'), t.timestamp.dt.hour.rename('hour')]).size().rename('posts').reset_index()
    a, b = st.columns(2)
    with a, guard(): style(px.density_heatmap(hm, x='hour', y='day', z='posts', color_continuous_scale='Purples', title='When users post (rows with real times)', category_orders={'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}))
    ua = f.groupby('user_id').agg(posts=('post_id', 'count'), eng=('eng', 'sum')).reset_index().merge(u, on='user_id')
    with b, guard(): style(px.scatter(ua, x='follower_count', y='eng', size='posts', color='language', hover_name='user_id', title=f'Followers vs engagement (r = {ua.follower_count.corr(ua.eng):.2f})'))
    a, b = st.columns(2)
    with a: style(px.bar(u.location.value_counts().head(12).reset_index(), x='count', y='location', orientation='h', title='Top user locations'))
    with b: style(px.pie(u.language.value_counts().reset_index(), names='language', values='count', hole=.5, title='User languages'))
with t3, guard("This section is temporarily unavailable."):
    flow(['Corrupted website', 'R1·P1 Clean & EDA', 'MySQL tables', 'R1·P2 SQL insights', 'R2 TF-IDF + LogReg', 'R3 Live data (Ars + HN)', 'R4 Dashboard'])
    st.dataframe(pd.DataFrame({'Stage': ['R1 P1', 'R1 P2', 'R2', 'R3', 'R4'],
        'Input': ['Corrupted posts and users CSV', 'Cleaned posts + users', 'Labelled text (9,000)', 'Reddit human-verification discussion', 'All outputs'],
        'Technique': ['Dedup, imputation, standardisation', 'Joins, CTEs, window functions', 'TF-IDF (1-2 grams) + Logistic Regression', 'API + HTML scraping, NLP scoring', 'Streamlit + Plotly'],
        'Output': ['12,000 clean posts', 'Platform, user, ranking insights', '65.2% accuracy, 3-class sentiment', '72 scored records', 'This dashboard']}), hide_index=True, use_container_width=True)
    note('Use the sidebar to move through the pages: <b>SQL Core</b> → <b>Semantic Layer</b> → <b>Live Monitor</b>. The Round 2 model is reused on Round 1 posts and on the Round 3 live data.')
