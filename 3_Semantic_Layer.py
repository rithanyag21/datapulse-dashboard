import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from common import *
setup('Semantic Layer', 'Round 2 — TF-IDF + Logistic Regression sentiment classifier (Negative · Neutral · Positive)')
CM = np.array([[418, 102, 80], [125, 356, 119], [64, 135, 400]]); L = ['Negative', 'Neutral', 'Positive']
kpis([('Accuracy', '65.2%', '1,800 test posts'), ('Macro F1', '0.652', 'balanced classes'), ('Correct / wrong', '1,174 / 626', ''), ('Polarity inversions', '144', '8.0% of test set')])
t1, t2, t3, t4 = st.tabs(['📈 Model comparison', '🧩 Confusion & errors', '⚡ Live predictor', '🔬 Reliability'])
with t1, guard("This section is temporarily unavailable."):
    m = pd.DataFrame({'Model': ['Logistic Regression', 'Linear SVM', 'Multinomial NB'], 'Accuracy': [.6522, .6517, .6378], 'F1': [.6520, .6515, .6353]})
    style(px.bar(m.melt('Model'), x='Model', y='value', color='variable', barmode='group', range_y=[.6, .67], title='Same TF-IDF features, same split'))
    pc = pd.DataFrame({'Class': L, 'Precision': [.654, .638, .6638], 'Recall': [.648, .6417, .6667], 'F1': [.651, .6398, .6652]})
    style(px.bar(pc.melt('Class'), x='Class', y='value', color='variable', barmode='group', range_y=[.6, .7], title='Per-class scores (Logistic Regression)'), 330)
    note('<b>Why Logistic Regression:</b> best accuracy and F1 on the split, plus calibrated class probabilities (used in the live predictor and Round 3) and fast inference. Its lead over Linear SVM is small.')
with t2, guard("This section is temporarily unavailable."):
    norm = st.toggle('Show row percentages', True)
    z = CM / CM.sum(1, keepdims=True) * 100 if norm else CM
    f = go.Figure(go.Heatmap(z=z, x=L, y=L, colorscale='Purples', text=np.round(z, 1 if norm else 0), texttemplate='%{text}', showscale=False))
    f.update_layout(title='Confusion matrix (rows = actual)', xaxis_title='Predicted', yaxis_title='Actual', yaxis_autorange='reversed'); style(f, 400)
    kpis([('Neutral ↔ Negative', 227, 'errors'), ('Neutral ↔ Positive', 254, 'errors'), ('Negative ↔ Positive', 144, 'inversions')])
    note('<b>Error pattern:</b> 76.8% of mistakes involve the Neutral class. Short posts (about 108 characters) give sparse bag-of-words features, so sarcasm and mixed tone are missed.')
    e = pd.read_csv(ROOT / 'data/error_analysis.csv'); a, b, c = st.columns([1, 1, 2])
    ac = a.selectbox('Actual', ['Any'] + L); pdn = b.selectbox('Predicted', ['Any'] + L); s = c.text_input('Search text')
    if ac != 'Any': e = e[e.actual_label == ac]
    if pdn != 'Any': e = e[e.predicted_label == pdn]
    if s: e = e[e.text.str.contains(s, case=False, na=False)]
    st.caption(f'{len(e)} misclassified posts'); st.dataframe(e, hide_index=True, use_container_width=True, height=280)
with t3, guard("This section is temporarily unavailable."):
    ex = {'Custom': '', 'Praise': 'Absolutely loving the new update, works great!', 'Complaint': 'Worst customer service ever, totally disappointed.', 'News-like': 'The company announced a new policy on Monday.'}
    ch = st.selectbox('Try an example', list(ex)); txt = st.text_area('Post text', ex[ch] or 'This verification step is a great idea to stop bots.', height=100)
    if txt.strip():
        cl, pr = predict([txt]); i = pr[0].argmax()
        st.markdown(f'### Prediction: <span style="color:{SC[cl[i]]}">{cl[i]}</span> ({pr[0][i]:.0%} confidence)', unsafe_allow_html=True)
        style(px.bar(x=cl, y=pr[0], color=cl, color_discrete_map=SC, labels={'x': '', 'y': 'probability'}), 300)
with t4, guard("This section is temporarily unavailable."):
    note('<b>Honest reading of the score:</b> 1,100 texts in the dataset are duplicates, and 17.4% of test posts also appear in training. On posts the model has never seen, accuracy is about 60% (about 0.59 macro F1 under duplicate-aware cross-validation). This is why the model is used as a trend signal and not as a per-post verdict.', warn=True)
    note('<b>Improvement path:</b> add character n-grams (about +2 points on the same split), remove duplicate texts before splitting, and fine-tune a transformer.')
