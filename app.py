# app.py
# Run: streamlit run app.py

import html
import re
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import llm

TITLE_TEXT = "Writing a Complete Scientific Argument"
st.set_page_config(page_title=TITLE_TEXT, layout="wide")

# ---------- Header & CSS ----------
st.markdown(f"""
<style>
/* Top header */
.app-header {{
  background: #0f2b3a; color: #fff; width: 100%;
  padding: 14px 24px; border-radius: 8px; margin: 8px 0 20px 0;
  box-shadow: 0 1px 2px rgba(0,0,0,.08);
}}
.app-header h1 {{ margin: 0; font-size: 22px; font-weight: 600; }}

/* General tweaks */
.stButton > button {{ white-space: nowrap; padding: 0.5rem 0.9rem; }}
.stDataFrame table thead tr th {{ white-space: nowrap !important; word-break: keep-all !important; }}

/* Bold entire radio labels */
[data-testid="stRadio"] label p {{ font-weight: 600 !important; }}

/* Instruction line */
.inst {{
  font-size: 1.05rem;
  color: #374151;
  margin: 4px 0 10px 2px;
  line-height: 1.4;
}}

/* Robot feedback bar with left color rail */
.fb{{
  display:flex; gap:12px; align-items:center;
  padding:14px 16px; border-radius:10px; border:1px solid;
  margin:12px 0 18px 0;
  font-size:1.06rem; line-height:1.5;
}}
.fb--ok  {{ background:#E8FAF0; border-color:#B7E2C2; color:#14532D; border-left:6px solid #16A34A; }}
.fb--warn{{ background:#FEF3C7; border-color:#FDE68A; color:#7C2D12; border-left:6px solid #F59E0B; }}

.fb__icon{{
  font-size:28px; line-height:1; min-width:28px;
  display:flex; align-items:center;
}}
.fb__text{{ flex:1; }}

/* History panels */
.history-wrap{{ border:1px solid #e5e7eb; border-radius:10px; padding:8px; }}
.history-scroll{{ max-height: 360px; overflow-y:auto; padding-right:8px; }}
.hist-card{{ border:1px solid #eef0f2; border-radius:10px; padding:10px 12px; margin:10px 0; background:#fff; }}
.hist-title{{ display:flex; justify-content:space-between; gap:8px; font-size:0.95rem; }}
.badge-pass{{ background:#e6f6ea; color:#166534; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge-fail{{ background:#fff7ed; color:#9a3412; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge-claim{{ background:#eff6ff; color:#1d4ed8; padding:2px 8px; border-radius:999px; }}
.smallnote{{ color:#6b7280; font-size:0.9rem; }}
.section-label{{ font-weight:600; color:#374151; margin-top:6px; }}
.hist-body blockquote{{ margin:6px 0; padding:6px 10px; background:#f8fafc; border-left:3px solid #cbd5e1; }}
.kv{{ display:grid; grid-template-columns: 160px 1fr; gap:6px 10px; margin:8px 0; }}
.kv div:nth-child(odd){{ color:#6b7280; }}
</style>
<div class="app-header"><h1>{TITLE_TEXT}</h1></div>
""", unsafe_allow_html=True)

# ---------- Small helpers ----------
def _esc_html(s: str) -> str:
    if not s:
        return ""
    return html.escape(s).replace("\n", "<br>")

def show_feedback_bar(text: str, passed: bool = True, who: str = "Tutor feedback"):
    icon_html = "🤖"
    klass = "fb--ok" if passed else "fb--warn"
    html_box = f"""
    <div class="fb {klass}">
      <div class="fb__icon">{icon_html}</div>
      <div class="fb__text"><strong>{who}:</strong> {_esc_html(text)}</div>
    </div>
    """
    st.markdown(html_box, unsafe_allow_html=True)

def show_feedback_turns(text: str, passed: bool = True, who: str = "Tutor feedback"):
    if not text:
        return

    pattern = re.compile(r"(Turn\s+\d+:)")
    matches = list(pattern.finditer(text))

    if not matches:
        show_feedback_bar(text, passed=passed, who=who)
        return

    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip()
        if seg:
            segments.append(seg)

    for seg in segments:
        seg_clean = re.sub(r"^Turn\s+\d+:\s*", "", seg, flags=re.IGNORECASE)
        show_feedback_bar(seg_clean, passed=passed, who=who)

def _passed_from_pred(pred: dict) -> bool:
    if not isinstance(pred, dict):
        return True
    ev = (pred.get("evidence") or "").lower()
    rs = (pred.get("reasoning") or "").lower()
    return ("supportive" in ev) and ("valid" in rs)

def _render_attempt_history(records, claim_label: str):
    if not records:
        st.caption("No attempts yet for this claim. Submit to see history here.")
        return

    html_cards = ['<div class="history-wrap"><div class="history-scroll">']
    for idx, r in enumerate(reversed(records), start=1):
        pred = r.get("predicted_labels") or {}
        passed = _passed_from_pred(pred)
        badge = "badge-pass" if passed else "badge-fail"
        status = "Passed" if passed else "Needs work"

        ev_text = r.get("evidence_text", "")
        rs_text = r.get("reasoning_text", "")
        fb_text = r.get("feedback", "")

        html_cards.append(
            f'<div class="hist-card">'
            f'  <div class="hist-title">'
            f'    <div><strong>Attempt #{len(records)-idx+1}</strong> • <span class="smallnote">{r.get("ts","")}</span></div>'
            f'    <div><span class="{badge}">{status}</span> <span class="badge-claim">{claim_label}</span></div>'
            f'  </div>'
            f'  <div class="kv">'
            f'    <div>Claim</div><div>{_esc_html(str(pred.get("claim","")) )}</div>'
            f'    <div>Evidence</div><div>{_esc_html(str(pred.get("evidence","")) )}</div>'
            f'    <div>Reasoning</div><div>{_esc_html(str(pred.get("reasoning","")) )}</div>'
            f'    <div>Pattern</div><div>{_esc_html(str(pred.get("reasoning_pattern","")) )}</div>'
            f'  </div>'
            f'  <div class="hist-body">'
            f'    <div class="section-label">Your evidence</div>'
            f'    <blockquote>{_esc_html(ev_text) if ev_text else "<i>(empty)</i>"}</blockquote>'
            f'    <div class="section-label">Your reasoning</div>'
            f'    <blockquote>{_esc_html(rs_text) if rs_text else "<i>(empty)</i>"}</blockquote>'
            f'    <div class="section-label">Feedback</div>'
            f'    <blockquote>{_esc_html(fb_text) if fb_text else "<i>(empty)</i>"}</blockquote>'
            f'  </div>'
            f'</div>'
        )

    html_cards.append('</div></div>')
    st.markdown("".join(html_cards), unsafe_allow_html=True)

# ---------- State ----------
def init_state():
    ss = st.session_state
    ss.setdefault("claim", None)
    ss.setdefault("prev_claim", None)
    ss.setdefault("claim_radio", None)

    ss.setdefault("evidence_text", "")
    ss.setdefault("reasoning_text", "")

    ss.setdefault("attempt_history", [])

    ss.setdefault("last_feedback", "")
    ss.setdefault("last_predicted", None)

    ss.setdefault("submitted", False)

init_state()

def reset_after_claim_change(keep_text=True):
    if not keep_text:
        st.session_state.evidence_text = ""
        st.session_state.reasoning_text = ""
    st.session_state.last_feedback = ""
    st.session_state.last_predicted = None
    st.session_state.submitted = False

# ---------- Data & Figure ----------
@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.DataFrame({
        "Year": [1, 2, 3, 4, 5],
        "# of Corn Planted": [130, 130, 130, 130, 130],
        "# of Corn Harvested": [130, 97, 91, 84, 80],
        "Harvest Spiders": [0, 0, 10, 10, 10],
        "Rootworms Eggs Initial": [0, 18, 29, 41, 41],
        "Rootworms Eggs Final": [0, 53, 89, 89, 100],
    })

@st.cache_resource(show_spinner=False)
def build_figure(df: pd.DataFrame):
    palette = ["#4C78A8", "#72B7B2", "#A0A0A0", "#F2CF5B", "#E15759"]
    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=120)
    x = df["Year"]
    ax.plot(x, df["# of Corn Planted"], marker="o", label="# of Corn Planted", color=palette[0])
    ax.plot(x, df["# of Corn Harvested"], marker="o", label="# of Corn Harvested", color=palette[1])
    ax.plot(x, df["Harvest Spiders"], marker="o", label="# Harvest Spiders", color=palette[2])
    ax.plot(x, df["Rootworms Eggs Initial"], marker="o", label="# Rootworm Eggs Initial", color=palette[3])
    ax.plot(x, df["Rootworms Eggs Final"], marker="o", label="# Rootworm Eggs Final", color=palette[4])
    ax.set_xlabel("Year"); ax.set_ylabel("Count"); ax.set_xlim(1,5); ax.grid(alpha=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout()
    return fig

df = load_dataset()
fig = build_figure(df)

# ---------- Layout ----------
left, right = st.columns([1.2, 1.8], gap="large")

with left:
    st.header("Task Overview")
    st.markdown(
        """
One of your classmates found **actual data** collected from a corn farm that was facing a rootworm infestation, just like the garden in your school. Based on the data, some of your classmates predict that continuing to add 10 harvest spiders will help improve the corn harvest in Year 6. **Do you agree or disagree with this prediction?**  
Analyze the data trend and make your own prediction about the Year 6 corn harvest. Your response should include a claim, supporting data, and valid reasoning.
        """
    )
    st.markdown("**Yearly data:**")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("**Impact of Harvest Spiders on Corn Harvest and Rootworm Population:**")
    st.pyplot(fig, use_container_width=True)

with right:
    st.header("Student Workspace")

    # Step 1 — Claim
    st.subheader("1) Choose your claim")
    agree_text = "I agree that continuing to add 10 harvest spiders will help improve the corn harvest in year 6."
    disagree_text = "I disagree that continuing to add 10 harvest spiders will help improve the corn harvest in year 6."
    initial_idx = None
    if st.session_state.claim_radio is None:
        if st.session_state.claim == "agree": initial_idx = 0
        elif st.session_state.claim == "disagree": initial_idx = 1

    claim_choice = st.radio(
        "Choose one",
        [agree_text, disagree_text],
        index=initial_idx,
        key="claim_radio",
        label_visibility="collapsed"
    )
    if claim_choice is None:
        st.session_state.claim = None
        st.info("Please choose your claim to continue.")
        st.stop()
    else:
        new_claim = "agree" if claim_choice == agree_text else "disagree"
        if st.session_state.prev_claim is not None and new_claim != st.session_state.prev_claim:
            st.session_state.claim = new_claim
            reset_after_claim_change(keep_text=True)
            st.info("Claim changed. Your previous feedback is cleared (text kept).")
        else:
            st.session_state.claim = new_claim
        st.session_state.prev_claim = new_claim
        st.success(f"Selected claim: **{st.session_state.claim.capitalize()}**")

    # Step 2 — Evidence
    st.divider(); st.subheader("2) Evidence")
    st.markdown('<div class="inst">Now present your evidence. Use the data from the table and chart to support your claim.</div>',
                unsafe_allow_html=True)
    st.text_area(
        "Evidence text",
        key="evidence_text",
        height=150,
        placeholder='Write your evidence here.',
        label_visibility="collapsed"
    )

    # Step 3 — Reasoning
    st.divider(); st.subheader("3) Reasoning")
    st.markdown('<div class="inst">Describe how your evidence supports your claim. Use what you know about predators, prey, and ecosystem balance to explain how your data supports your claim.</div>',
                unsafe_allow_html=True)
    st.text_area(
        "Reasoning text",
        key="reasoning_text",
        height=160,
        placeholder='Write your reasoning here.',
        label_visibility="collapsed"
    )

    # Unified feedback
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        run_btn = st.button("Get feedback (Evidence + Reasoning)", type="primary", use_container_width=True)
    with c2:
        if st.button("Clear feedback", use_container_width=True):
            st.session_state.last_feedback = ""
            st.session_state.last_predicted = None
            st.rerun()

    if run_btn:
        try:
            curr_claim = st.session_state.claim

            # Build chat_history for this claim with round_index
            prior = []
            claim_count = 0
            for r in st.session_state.attempt_history:
                if r.get("claim_choice") == curr_claim:
                    claim_count += 1
                    prior.append({
                        "round_index": claim_count,
                        "student_resp": r.get("student_resp", ""),
                        "predicted_labels": r.get("predicted_labels", {}),
                        "evidence_info": r.get("evidence_info"),
                        "reasoning_info": r.get("reasoning_info"),
                        "feedback": r.get("feedback", ""),
                    })

            with st.spinner("Scoring your evidence + reasoning…"):
                result = llm.evaluate_round(
                    claim_choice=curr_claim,
                    evidence_text=st.session_state.evidence_text,
                    reasoning_text=st.session_state.reasoning_text,
                    chat_history=prior,
                )

            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "claim_choice": curr_claim,
                "evidence_text": (st.session_state.evidence_text or "").strip(),
                "reasoning_text": (st.session_state.reasoning_text or "").strip(),
                "predicted_labels": result.get("predicted_labels"),
                "evidence_info": result.get("evidence_info"),
                "reasoning_info": result.get("reasoning_info"),
                "feedback": result.get("feedback", "").strip(),
                "student_resp": result.get("student_resp", ""),
            }
            st.session_state.attempt_history.append(record)

            st.session_state.last_feedback = record["feedback"]
            st.session_state.last_predicted = record["predicted_labels"]
            st.session_state.submitted = False

            st.rerun()

        except AttributeError:
            st.error("llm.evaluate_round(...) not found. Check that llm.py is updated.")
        except Exception as e:
            st.error(f"Error while scoring: {e}")

    # Show latest feedback
    if st.session_state.last_feedback:
        passed = _passed_from_pred(st.session_state.last_predicted or {})
        show_feedback_turns(st.session_state.last_feedback, passed=passed)

    # Show latest classification
    if st.session_state.last_predicted:
        st.markdown("**Classification Results (latest)**")
        st.json(st.session_state.last_predicted)

    # History (current claim only)
    st.markdown("**Your Attempts & Feedback**")
    curr_claim = st.session_state.claim
    records = [r for r in st.session_state.attempt_history if r.get("claim_choice") == curr_claim]
    other = len(st.session_state.attempt_history) - len(records)
    if other > 0:
        st.caption(f"{other} attempt(s) from the other claim are hidden.")
    _render_attempt_history(records, curr_claim.capitalize())

    # Final submit
    st.divider()
    if st.button("Submit", type="secondary", use_container_width=True):
        st.session_state.submitted = True
        st.success("✅ Submitted! Your response has been recorded.")
        st.balloons()

    if st.session_state.submitted:
        st.caption("Thanks! You can still revise and get feedback above.")