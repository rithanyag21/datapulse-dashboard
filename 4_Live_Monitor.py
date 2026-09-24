import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from common import *
setup('Live Monitor', "Round 3 — Reddit's human-verification announcement: Ars OpenForum + Hacker News reactions, scored by the Round 2 model")
d = r3(); c1, c2 = st.columns(2)
src = c1.multiselect('Source', sorted(d.source.unique()), default=sorted(d.source.unique()))
ms = c2.multiselect('Round 2 model sentiment', ['Negative', 'Neutral', 'Positive'], default=['Negative', 'Neutral', 'Positive'])
f = d[d.source.isin(src) & d.r2_pred.isin(ms)]
if f.empty: st.warning('No records match the filters.'); st.stop()
kpis([('Records', len(f), f'{(f.source == "Hacker News").sum()} HN · {(f.source == "Ars OpenForum").sum()} Ars'), ('Model: Negative', f'{(f.r2_pred == "Negative").mean():.0%}', 'Round 2 model'),
      ('Lexicon: Positive', f'{(f.sentiment == "Positive").mean():.0%}', 'VADER-style scores'), ('Agreement', f'{(f.r2_pred == f.sentiment).mean():.0%}', 'model vs lexicon'), ('Avg words', f'{f.word_count.mean():.0f}', 'per post')])
t1, t2, t3, t4 = st.tabs(['🎭 Sentiment', '⏱ Activity & shifts', '🏷 Topics & entities', '🗂 Data explorer'])
with t1, guard("This section is temporarily unavailable."):
    a, b = st.columns(2)
    with a: style(px.pie(f.r2_pred.value_counts().reset_index(), names='r2_pred', values='count', hole=.55, color='r2_pred', color_discrete_map=SC, title='Round 2 model (primary)'))
    with b: style(px.pie(f.sentiment.value_counts().reset_index(), names='sentiment', values='count', hole=.55, color='sentiment', color_discrete_map=SC, title='Lexicon scores (comparison)'))
    ct = pd.crosstab(f.sentiment, f.r2_pred)
    style(px.imshow(ct, text_auto=True, color_continuous_scale='Purples', labels=dict(x='Round 2 model', y='Lexicon'), title='Where the two methods disagree'), 330)
    note('<b>Domain shift:</b> the Round 2 model was trained on general social text and labels most forum comments Negative. The lexicon reads the same posts as mostly Positive. Both views are shown so the disagreement is visible, not hidden. Treat the model output as relative signal.', warn=True)
with t2, guard("This section is temporarily unavailable."):
    g = f.groupby('thread_date').agg(records=('post_id', 'count'), lexicon=('compound_score', 'mean'), model_neg=('r2_pred', lambda s: (s == 'Negative').mean())).reset_index()
    a, b = st.columns(2)
    with a: style(px.bar(g, x='thread_date', y='records', text='records', title='Discussion volume by thread date'))
    with b:
        fg = go.Figure([go.Scatter(x=g.thread_date, y=g.lexicon, name='Lexicon compound', mode='lines+markers', line_color=P['ora']), go.Scatter(x=g.thread_date, y=g.model_neg, name='Model share Negative', mode='lines+markers', yaxis='y2', line_color=P['neg'])])
        fg.update_layout(title='Sentiment by date', yaxis2=dict(overlaying='y', side='right', range=[0, 1])); style(fg)
    st.markdown('#### Detected changes')
    for r0, r1 in zip(g.itertuples(), g.iloc[1:].itertuples()):
        note(f'<b>Shift {r0.thread_date} → {r1.thread_date}:</b> lexicon compound {r0.lexicon:+.2f} → {r1.lexicon:+.2f} · model Negative share {r0.model_neg:.0%} → {r1.model_neg:.0%}')
    top = g.loc[g.records.idxmax()]
    note(f'<b>Engagement spike:</b> {top.thread_date} holds {int(top.records)} of {len(f)} records ({top.records / len(f):.0%}). <b>Trigger:</b> the Ars Technica report that Reddit will require "fishy" accounts to verify they are human.')
    note('<b>Data limits:</b> only the 8 Hacker News comments have verified timestamps. The 64 Ars replies are placed on the thread date (2026-03-25), and engagement counts were not exposed, so the spike reflects volume, not likes. Shifts across dates rest on very few posts.', warn=True)
with t3, guard("This section is temporarily unavailable."):
    e = f.explode('themes').dropna(subset=['themes']).reset_index(drop=True)
    a, b = st.columns(2)
    with a, guard(): style(px.bar(e.themes.value_counts().reset_index(), x='count', y='themes', orientation='h', title='Discussion themes'))
    with b:
        n = st.slider('Top terms', 8, 30, 15); cv = CountVectorizer(stop_words='english', min_df=2, token_pattern=r'[A-Za-z]{3,}')
        w = pd.Series(cv.fit_transform(f.clean_text).sum(0).A1, cv.get_feature_names_out()).nlargest(n).reset_index(); w.columns = ['term', 'count']
        style(px.bar(w, x='count', y='term', orientation='h', title='Most frequent terms'))
    with guard():
      if len(e): style(px.bar(pd.crosstab(e.themes, e.r2_pred).reset_index().melt('themes'), x='themes', y='value', color='r2_pred', color_discrete_map=SC, title='Model sentiment by theme'))
with t4, guard("This section is temporarily unavailable."):
    q = st.text_input('Search posts'); v = f[f.clean_text.str.contains(q, case=False, na=False)] if q else f
    st.dataframe(v[['source', 'thread_date', 'r2_pred', 'r2_conf', 'sentiment', 'compound_score', 'clean_text', 'url']], hide_index=True, use_container_width=True, height=420)
    st.download_button('⬇ Download scored dataset', v.drop(columns='themes').to_csv(index=False), 'datapulse_r3_scored.csv')
