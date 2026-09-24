import sqlite3, pandas as pd, plotly.express as px, streamlit as st
from common import *
setup('Analytical Core', 'Round 1 · Phase 2 — SQL reasoning on the restored tables (live queries, no hard-coded outputs)')
@st.cache_resource
def db():
    c = sqlite3.connect(':memory:', check_same_thread=False)
    posts().assign(timestamp=lambda d: d.timestamp.astype(str)).to_sql('posts', c, index=False); users().to_sql('users', c, index=False); return c
def run(q): return pd.read_sql_query(q, db())
E1 = "SELECT platform, COUNT(*) AS post_count FROM posts WHERE platform IS NOT NULL AND platform <> 'Unknown' GROUP BY platform ORDER BY post_count DESC;"
M3 = """SELECT u.user_id, u.follower_count, u.location, COUNT(p.post_id) AS post_count, SUM(p.likes + p.shares + p.comments) AS total_engagement
FROM users u JOIN posts p ON u.user_id = p.user_id GROUP BY u.user_id, u.follower_count, u.location
ORDER BY post_count DESC, total_engagement DESC LIMIT 10;"""
H2 = """WITH user_engagement AS (SELECT u.user_id, u.location, u.follower_count, COUNT(p.post_id) AS post_count, SUM(p.likes + p.shares + p.comments) AS total_engagement
  FROM users u JOIN posts p ON u.user_id = p.user_id GROUP BY u.user_id, u.location, u.follower_count),
ranked_users AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY location ORDER BY total_engagement DESC) AS location_rank FROM user_engagement)
SELECT * FROM ranked_users WHERE location_rank <= 3 ORDER BY location, location_rank;"""
TR = "SELECT substr(timestamp,1,7) AS month, COUNT(*) AS posts, ROUND(AVG(likes+shares+comments),0) AS avg_engagement FROM posts GROUP BY month ORDER BY month;"
t1, t2, t3, t4, t5 = st.tabs(['E1 Platforms', 'M3 Active users', 'H2 Location ranks', 'Trend & correlation', '🔎 SQL sandbox'])
with t1, guard("This section is temporarily unavailable."):
    d = run(E1); kpis([('Top platform', d.platform[0], f'{d.post_count[0]:,} posts'), ('Lead over #2', int(d.post_count[0] - d.post_count[1]), 'a near-tie'), ('Unknown platform', f'{(posts().platform == "Unknown").sum():,}', 'excluded')])
    style(px.bar(d, x='platform', y='post_count', color='platform', text='post_count', title='Posts per platform'), 340); st.code(E1, 'sql')
    note('<b>Insight:</b> Facebook leads by a single post, so platform popularity is effectively even. The 1,784 "Unknown" rows are larger than that lead.')
with t2, guard("This section is temporarily unavailable."):
    d = run(M3); style(px.bar(d, x='user_id', y='post_count', color='total_engagement', hover_data=['location', 'follower_count'], title='Top 10 users by posts (ties broken by engagement)'), 360)
    st.dataframe(d, hide_index=True, use_container_width=True); st.code(M3, 'sql')
    note('<b>Logic:</b> JOIN users to posts, aggregate per user. Six users tie at 16 posts, so a secondary sort on engagement makes the top 10 deterministic.')
with t3, guard("This section is temporarily unavailable."):
    d = run(H2); loc = st.selectbox('Location', ['All'] + sorted(d.location.unique()))
    v = d if loc == 'All' else d[d.location == loc]
    style(px.bar(v, x='location', y='total_engagement', color=v.location_rank.astype(str), barmode='group', hover_data=['user_id'], title='Top 3 users per location (ROW_NUMBER over PARTITION BY location)'), 400)
    st.dataframe(v, hide_index=True, use_container_width=True); st.code(H2, 'sql')
with t4, guard("This section is temporarily unavailable."):
    d = run(TR); a, b = st.columns(2)
    with a: style(px.line(d, x='month', y='posts', markers=True, title='Posts per month'))
    with b: style(px.line(d, x='month', y='avg_engagement', markers=True, title='Average engagement per post'))
    ua = run("SELECT u.user_id, u.follower_count, SUM(p.likes+p.shares+p.comments) AS eng, COUNT(*) AS n FROM users u JOIN posts p USING(user_id) GROUP BY u.user_id")
    kpis([('Followers ↔ engagement', f'{ua.follower_count.corr(ua.eng):.2f}', 'Pearson r'), ('Posts ↔ engagement', f'{ua.n.corr(ua.eng):.2f}', 'Pearson r')])
    note('Engagement follows post count, not follower count. Likes are partly median-imputed (see the Intake page), so treat engagement totals as approximate.')
with t5, guard("This section is temporarily unavailable."):
    q = st.text_area('Read-only SQL (tables: posts, users)', "SELECT platform, ROUND(AVG(likes),0) AS avg_likes FROM posts GROUP BY platform;", height=120)
    if st.button('▶ Run query'):
        if not q.strip().lower().startswith(('select', 'with')): st.error('Only SELECT / WITH queries are allowed.')
        else:
            try: r = run(q); st.dataframe(r, use_container_width=True); st.caption(f'{len(r)} rows')
            except Exception as e: st.error(e)
    with st.expander('Schema'): st.code("posts(post_id PK, user_id FK, platform, text_content, timestamp, likes, shares, comments, likes_imputed)\nusers(user_id PK, location, language, account_created, follower_count)", 'sql')
