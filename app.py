import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import re
import json
import matplotlib.pyplot as plt

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Modern & Clean Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    .stApp {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Headers */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        color: #1e293b;
        margin-bottom: 0.8rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.18rem;
        margin-bottom: 2.5rem;
    }

    /* Cards & Results */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        background: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    .fake-result   { border-top: 6px solid #ef4444; }
    .real-result   { border-top: 6px solid #10b857; }
    .susp-result   { border-top: 6px solid #f59e0b; }

    .confidence-big {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.8rem 0;
    }

    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        text-align: center;
        border: 1px solid #f1f5f9;
    }

    /* Report items */
    .report-item {
        background: white;
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        padding: 0.65rem 1.3rem;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: #3b82f6;
        color: white;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ─── Mock Firebase ───────────────────────────────────────────────────────────
class FirebaseManager:
    def __init__(self):
        self.data = {
            'reports': [],
            'stats': {
                'total_checks': 0,
                'fake_detected': 0,
                'real_detected': 0,
                'suspicious_detected': 0,
                'user_reports': 0
            }
        }

    def add_report(self, title, content, prediction, confidence, features):
        rid = hashlib.md5(f"{title}{datetime.now()}".encode()).hexdigest()[:10]
        report = {
            'id': rid,
            'title': title,
            'content': content[:480] + '…' if len(content) > 480 else content,
            'prediction': prediction,
            'confidence': confidence,
            'features': features,
            'timestamp': datetime.now().isoformat(),
        }
        self.data['reports'].append(report)
        self.data['stats']['total_checks'] += 1
        if prediction == 'FAKE':
            self.data['stats']['fake_detected'] += 1
        elif prediction == 'REAL':
            self.data['stats']['real_detected'] += 1
        elif prediction == 'SUSPICIOUS':
            self.data['stats']['suspicious_detected'] += 1
        return rid

    def add_user_feedback(self, report_id, feedback_type):
        self.data['stats']['user_reports'] += 1

    def get_stats(self):
        return self.data['stats']

    def get_recent_reports(self, limit=12):
        sorted_reports = sorted(
            self.data['reports'],
            key=lambda x: x.get('timestamp', '2000-01-01T00:00:00Z'),
            reverse=True
        )
        return sorted_reports[:limit]

firebase = FirebaseManager()

# ─── Detector Logic ──────────────────────────────────────────────────────────
class FakeNewsDetector:
    def extract_features(self, text: str) -> dict:
        if not text:
            return {}
        text = text.lower()
        f = {}
        f['char_count'] = len(text)
        f['word_count'] = len(text.split())
        clickbait_patterns = [
            r'breaking.*exclusive', r'you won\'?t believe', r'shocking.*revealed',
            r'they don\'?t want you to know', r'doctors hate', r'click.*here',
            r'limited time', r'secret.*government', r'hidden truth', r'must (see|watch)',
            r'viral.*video', r'miracle.*cure'
        ]
        f['suspicious_patterns'] = sum(bool(re.search(p, text)) for p in clickbait_patterns)
        emotional = [
            'amazing', 'horrible', 'terrible', 'unbelievable', 'shocking',
            'outrageous', 'fantastic', 'miracle', 'deadly', 'urgent', 'alert'
        ]
        f['emotional_words'] = sum(w in text for w in emotional)
        f['exclamation_count'] = text.count('!')
        f['all_caps_count'] = sum(1 for w in text.split() if w.isupper() and len(w) > 3)
        credible = ['reuters', 'bbc', 'ap ', 'npr', 'wall street journal', 'new york times']
        f['credible_mention'] = any(s in text for s in credible)
        return f

    def predict(self, title: str, content: str):
        text = f"{title} {content}"
        feats = self.extract_features(text)
        score = 0
        if feats.get('suspicious_patterns', 0) > 2: score += 30
        if feats.get('emotional_words', 0) > 4: score += 20
        if feats.get('exclamation_count', 0) > 4: score += 15
        if feats.get('all_caps_count', 0) > 2: score += 12
        if feats.get('credible_mention', False): score -= 30
        if feats.get('word_count', 0) > 220: score -= 18
        score = max(0, min(100, score))
        if score > 68:
            return "FAKE", score, feats
        if score > 42:
            return "SUSPICIOUS", score, feats
        return "REAL", 100 - score, feats

detector = FakeNewsDetector()

# ─── Sample Articles ─────────────────────────────────────────────────────────
SAMPLES = {
    1: {
        "title": "BREAKING: Miracle Cure Discovered – Big Pharma Panicking!",
        "content": "A revolutionary natural cure heals everything in days! Doctors hate this trick. Click now before it's banned!"
    },
    2: {
        "title": "New mRNA Cancer Therapy Shows Strong Early Results",
        "content": "According to Reuters and Nature Medicine, a phase II trial of a novel mRNA-based therapy demonstrated promising tumor reduction in 62% of participants..."
    },
    3: {
        "title": "GOVERNMENT CONFIRMS ALIENS – SHOCKING AREA 51 FOOTAGE!!!",
        "content": "Leaked documents PROVE aliens are real! You won't believe what they found!!! Watch before they delete this video!!!"
    }
}

def load_sample(num: int):
    if num in SAMPLES:
        return SAMPLES[num]["title"], SAMPLES[num]["content"]
    return "", ""

# ─── Main Application ────────────────────────────────────────────────────────
def main():
    if "title" not in st.session_state:
        st.session_state.title = ""
    if "content" not in st.session_state:
        st.session_state.content = ""

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🛡️ Fake News Scanner")
        st.info("Educational • Rule-based • Mock storage", icon="ℹ️")
        
        st.markdown("### Live Statistics")
        stats = firebase.get_stats()
        
        cols = st.columns(2)
        cols[0].metric("Total Checks", stats["total_checks"])
        cols[1].metric("User Feedback", stats["user_reports"])
        
        st.metric("Fake Detected", stats["fake_detected"], delta_color="inverse")
        st.metric("Real Detected", stats["real_detected"])
        st.metric("Suspicious", stats.get("suspicious_detected", 0))

        st.markdown("---")
        st.caption("**Typical fake news signals**:\n"
                   "• Clickbait titles & phrases\n"
                   "• Heavy emotional language\n"
                   "• Excessive punctuation\n"
                   "• ALL CAPS SHOUTING\n"
                   "• No credible sources mentioned")

    # ── Main Header ──────────────────────────────────────────────────────────
    st.markdown('<div class="main-header">Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Rule-based misinformation scanner • Educational tool</div>', unsafe_allow_html=True)

    tab_detect, tab_dash, tab_hist, tab_about = st.tabs(
        ["🛡️  Check Article", "📊  Dashboard", "🗂️  History", "ℹ️  About"]
    )

    # ── Tab 1: Detect ────────────────────────────────────────────────────────
    with tab_detect:
        st.subheader("Analyze News Article")

        col_input, col_controls = st.columns([6, 2.5])

        with col_input:
            st.session_state.title = st.text_input(
                "News Headline",
                value=st.session_state.title,
                placeholder="Paste or write the article headline...",
                key="title_input"
            )

            st.session_state.content = st.text_area(
                "Article Content",
                value=st.session_state.content,
                height=280,
                placeholder="Paste the full article text here...\nLonger text → better analysis",
                key="content_input"
            )

        with col_controls:
            st.markdown("**Quick samples**")
            if st.button("Sample 1 – Clickbait", use_container_width=True, key="s1"):
                st.session_state.title, st.session_state.content = load_sample(1)
                st.rerun()

            if st.button("Sample 2 – Credible", use_container_width=True, key="s2"):
                st.session_state.title, st.session_state.content = load_sample(2)
                st.rerun()

            if st.button("Sample 3 – Conspiracy", use_container_width=True, key="s3"):
                st.session_state.title, st.session_state.content = load_sample(3)
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Clear", use_container_width=True, key="clear"):
                st.session_state.title = ""
                st.session_state.content = ""
                st.rerun()

        st.markdown("---")

        if st.button("🔍 Analyze Article", type="primary", use_container_width=True, key="analyze"):
            if not st.session_state.title.strip() or not st.session_state.content.strip():
                st.error("Please fill in both title and content.")
            else:
                with st.spinner("Analyzing article..."):
                    label, score, features = detector.predict(
                        st.session_state.title,
                        st.session_state.content
                    )

                    # Result card
                    if label == "FAKE":
                        cls = "fake-result"
                        emoji = "🚨"
                        text = "Likely FAKE NEWS"
                        color = "#ef4444"
                    elif label == "SUSPICIOUS":
                        cls = "susp-result"
                        emoji = "⚠️"
                        text = "SUSPICIOUS Content"
                        color = "#f59e0b"
                    else:
                        cls = "real-result"
                        emoji = "✅"
                        text = "Likely CREDIBLE"
                        color = "#10b857"

                    st.markdown(f"""
                    <div class="result-card {cls}">
                        <h2 style="color:{color}; margin:0;">{emoji} {text}</h2>
                        <div class="confidence-big" style="color:{color};">{score:.0f}%</div>
                        <div style="color:#64748b; font-size:1.1rem;">confidence score</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("Key Indicators")
                    cols = st.columns(3)

                    indicators = [
                        ("Suspicious phrases", features.get("suspicious_patterns", 0), "🚩"),
                        ("Emotional words", features.get("emotional_words", 0), "😮"),
                        ("Exclamation marks", features.get("exclamation_count", 0), "❗"),
                        ("Shouting (ALL CAPS)", features.get("all_caps_count", 0), "🔠"),
                        ("Credible sources", "Yes" if features.get("credible_mention") else "No", "📰"),
                        ("Word count", features.get("word_count", 0), "📄"),
                    ]

                    for i, (name, value, icon) in enumerate(indicators):
                        with cols[i % 3]:
                            st.markdown(f'<div class="metric-card"><strong>{icon} {name}</strong><br><h3>{value}</h3></div>', unsafe_allow_html=True)

                    rid = firebase.add_report(
                        st.session_state.title,
                        st.session_state.content,
                        label,
                        score,
                        features
                    )

                    st.success(f"Analysis saved (ID: **{rid}**)")

                    st.markdown("**Was this analysis accurate?**")
                    fb1, fb2 = st.columns(2)
                    with fb1:
                        if st.button("👍 Yes, correct", key=f"good_{rid}"):
                            firebase.add_user_feedback(rid, "correct")
                            st.success("Thank you!")
                    with fb2:
                        if st.button("👎 No, incorrect", key=f"bad_{rid}"):
                            firebase.add_user_feedback(rid, "wrong")
                            st.info("Feedback noted. Thank you.")

    # ── Tab 2: Dashboard ─────────────────────────────────────────────────────
    with tab_dash:
        st.subheader("Analysis Overview")

        stats = firebase.get_stats()
        cols = st.columns(4)
        cols[0].metric("Total Analyses", stats["total_checks"])
        cols[1].metric("Fake Detected", stats["fake_detected"])
        cols[2].metric("Real Detected", stats["real_detected"])
        cols[3].metric("Suspicious", stats.get("suspicious_detected", 0))

        if stats["total_checks"] > 0:
            fig, ax = plt.subplots(figsize=(5.8, 5.2))
            labels = ["Fake", "Real", "Suspicious"]
            values = [
                stats["fake_detected"],
                stats["real_detected"],
                stats.get("suspicious_detected", 0)
            ]
            colors = ["#ef4444", "#10b857", "#f59e0b"]

            if sum(values) == 0:
                values = [1,1,1]

            ax.pie(values, labels=labels, autopct="%1.1f%%",
                   colors=colors, shadow=True, startangle=30,
                   textprops={"fontsize": 11}, pctdistance=0.82)
            ax.set_title("Result Distribution", fontsize=14, pad=20)
            st.pyplot(fig)

    # ── Tab 3: History ───────────────────────────────────────────────────────
    with tab_hist:
        st.subheader("Recent Analyses")

        reports = firebase.get_recent_reports(20)
        if not reports:
            st.info("No analyses yet. Try checking some articles!")
        else:
            for rep in reports:
                color = {
                    "FAKE": "#ef4444",
                    "REAL": "#10b857",
                    "SUSPICIOUS": "#f59e0b"
                }.get(rep["prediction"], "#6b7280")

                st.markdown(f"""
                <div class="report-item">
                    <strong>{rep['title'][:90]}{'…' if len(rep['title']) > 90 else ''}</strong><br>
                    <span style="color:{color}; font-weight:700;">{rep['prediction']} – {rep['confidence']:.0f}%</span><br>
                    <small style="color:#64748b;">{rep['timestamp'][:19]}</small>
                </div>
                """, unsafe_allow_html=True)

    # ── Tab 4: About ─────────────────────────────────────────────────────────
    with tab_about:
        st.subheader("About this Tool")
        st.markdown("""
        This is an **educational rule-based fake news indicator** (not a trained ML model).

        **What it looks for:**
        - Clickbait & sensational phrases
        - Overuse of emotional/outrage language
        - Excessive exclamation marks & ALL CAPS
        - Absence of credible source mentions
        - Very short or suspiciously long articles

        **Important limitations:**
        - Can produce false positives and false negatives
        - Should **not** be used as the only source of truth
        - Always cross-check with established fact-checking websites

        Built with **Streamlit** • Mock storage • For learning & demonstration purposes only.
        """)

    st.markdown("---")
    st.caption("🛡️ Fake News Detector • Educational mock version • 2025")

if __name__ == "__main__":
    main()