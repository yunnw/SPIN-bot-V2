# app.py
# Run: streamlit run app.py

import html
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
  font-size: 1.05rem;           /* ≈16–17px */
  color: #374151;
  margin: 4px 0 10px 2px;
  line-height: 1.4;
}}

/* Robot feedback bar with left color rail */
.fb{{
  display:flex; gap:12px; align-items:center;
  padding:14px 16px; border-radius:10px; border:1px solid;
  margin:12px 0 18px 0;             /* separate from history */
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
.history-scroll{{ max-height: 320px; overflow-y:auto; padding-right:8px; }}
.hist-card{{ border:1px solid #eef0f2; border-radius:10px; padding:10px 12px; margin:10px 0; background:#fff; }}
.hist-title{{ display:flex; justify-content:space-between; gap:8px; font-size:0.95rem; }}
.badge-pass{{ background:#e6f6ea; color:#166534; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge-fail{{ background:#fff7ed; color:#9a3412; padding:2px 8px; border-radius:999px; font-weight:600; }}
.badge-claim{{ background:#eff6ff; color:#1d4ed8; padding:2px 8px; border-radius:999px; }}
.hist-body blockquote{{ margin:6px 0; padding:6px 10px; background:#f8fafc; border-left:3px solid #cbd5e1; }}
.smallnote{{ color:#6b7280; font-size:0.9rem; }}
.section-label{{ font-weight:600; color:#374151; margin-top:6px; }}
</style>
<div class="app-header"><h1>{TITLE_TEXT}</h1></div>
""", unsafe_allow_html=True)

# ---------- Small helpers ----------
def _esc_html(s: str) -> str:
    if not s: return ""
    return html.escape(s).replace("\n", "<br>")

def show_feedback_bar(text: str, passed: bool = True, who: str = "Tutor feedback"):
    """Render a callout feedback bar with robot icon and left color rail."""
    icon_html = "🤖"
    klass = "fb--ok" if passed else "fb--warn"
    html_box = f"""
    <div class="fb {klass}">
      <div class="fb__icon">{icon_html}</div>
      <div class="fb__text"><strong>{who}:</strong> {_esc_html(text)}</div>
    </div>
    """
    st.markdown(html_box, unsafe_allow_html=True)

def _render_evidence_history(records, claim_label):
    if not records:
        st.caption("No attempts yet for this claim. Submit to see history here.")
        return
    html_cards = ['<div class="history-wrap"><div class="history-scroll">']
    for idx, r in enumerate(reversed(records), start=1):
        badge = "badge-pass" if r["passed"] else "badge-fail"
        status = "Passed" if r["passed"] else "Needs work"
        conf_txt = f'{r.get("confidence",0):.2f}' if isinstance(r.get("confidence"), (int,float)) else ""
        label_txt = r.get("label","")
        html_cards.append(
            f'<div class="hist-card">'
            f'  <div class="hist-title">'
            f'    <div><strong>Attempt #{len(records)-idx+1}</strong> • <span class="smallnote">{r["ts"]}</span></div>'
            f'    <div><span class="{badge}">{status}</span> <span class="badge-claim">{claim_label}</span></div>'
            f'  </div>'
            f'  <div class="smallnote">Label: {label_txt}{(" • Conf: "+conf_txt) if conf_txt else ""}</div>'
            f'  <div class="hist-body">'
            f'    <div class="section-label">Your claim</div>'
            f'    <blockquote>{_esc_html(claim_label)}</blockquote>'
            f'    <div class="section-label">Your evidence</div>'
            f'    <blockquote>{_esc_html(r["text"]) if r["text"] else "<i>(empty)</i>"}</blockquote>'
            f'    <div class="section-label">Feedback</div>'
            f'    <blockquote>{_esc_html(r["feedback"])}</blockquote>'
            f'  </div>'
            f'</div>'
        )
    html_cards.append('</div></div>')
    st.markdown("".join(html_cards), unsafe_allow_html=True)

def _render_reasoning_history(records, claim_label):
    if not records:
        st.caption("No attempts yet for this claim. Submit to see history here.")
        return
    html_cards = ['<div class="history-wrap"><div class="history-scroll">']
    for idx, r in enumerate(reversed(records), start=1):
        badge = "badge-pass" if r["passed"] else "badge-fail"
        status = "Passed" if r["passed"] else "Needs work"
        conf_txt = f'{r.get("confidence",0):.2f}' if isinstance(r.get("confidence"), (int,float)) else ""
        label_txt = r.get("label","")
        ev_snap = r.get("evidence", "")
        html_cards.append(
            f'<div class="hist-card">'
            f'  <div class="hist-title">'
            f'    <div><strong>Attempt #{len(records)-idx+1}</strong> • <span class="smallnote">{r["ts"]}</span></div>'
            f'    <div><span class="{badge}">{status}</span> <span class="badge-claim">{claim_label}</span></div>'
            f'  </div>'
            f'  <div class="smallnote">Label: {label_txt}{(" • Conf: "+conf_txt) if conf_txt else ""}</div>'
            f'  <div class="hist-body">'
            f'    <div class="section-label">Your claim</div>'
            f'    <blockquote>{_esc_html(claim_label)}</blockquote>'
            f'    <div class="section-label">Your evidence (snapshot)</div>'
            f'    <blockquote>{_esc_html(ev_snap) if ev_snap else "<i>(empty)</i>"}</blockquote>'
            f'    <div class="section-label">Your reasoning</div>'
            f'    <blockquote>{_esc_html(r["text"]) if r["text"] else "<i>(empty)</i>"}</blockquote>'
            f'    <div class="section-label">Feedback</div>'
            f'    <blockquote>{_esc_html(r["feedback"])}</blockquote>'
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

    ss.setdefault("evidence_ok", False)
    ss.setdefault("reasoning_ok", False)
    ss.setdefault("evidence_fb", "")
    ss.setdefault("reasoning_fb", "")

    ss.setdefault("submitted", False)

    ss.setdefault("evidence_history", [])
    ss.setdefault("reasoning_history", [])
init_state()

def reset_after_claim_change(keep_text=True):
    if not keep_text:
        st.session_state.evidence_text = ""
        st.session_state.reasoning_text = ""
    st.session_state.evidence_ok = False
    st.session_state.reasoning_ok = False
    st.session_state.evidence_fb = ""
    st.session_state.reasoning_fb = ""
    st.session_state.submitted = False

def unlock_evidence():
    st.session_state.evidence_ok = False
    st.session_state.evidence_fb = ""
    st.session_state.reasoning_ok = False
    st.session_state.reasoning_fb = ""
    st.session_state.submitted = False

def unlock_reasoning():
    st.session_state.reasoning_ok = False
    st.session_state.reasoning_fb = ""
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

# ---------- GPT helpers ----------
def gpt_eval_evidence(claim: str, text: str):
    with st.spinner("Scoring evidence…"):
        out = llm.step_feedback("evidence", claim, text)
    label = (out.get("label") or "").lower()
    if label not in llm.EVIDENCE_LABELS:
        raise RuntimeError(f"Unexpected evidence label: {label}")
    passed = (label == "supportive")
    fb = out.get("step_feedback") or ("OK" if passed else "Please refine your evidence.")
    conf = 0.0
    try:
        conf = float(out.get("confidence", 0) or 0)
    except Exception:
        pass
    return {"passed": passed, "feedback": fb, "label": label, "confidence": conf}

def gpt_eval_reasoning(claim: str, text: str):
    with st.spinner("Scoring reasoning…"):
        out = llm.step_feedback("reasoning", claim, text, evidence_text=st.session_state.evidence_text)
    label = (out.get("label") or "").lower()
    if label not in llm.REASONING_LABELS:
        raise RuntimeError(f"Unexpected reasoning label: {label}")
    passed = (label == "valid")
    fb = out.get("step_feedback") or ("OK" if passed else "Please refine your reasoning.")
    conf = 0.0
    try:
        conf = float(out.get("confidence", 0) or 0)
    except Exception:
        pass
    return {"passed": passed, "feedback": fb, "label": label, "confidence": conf}

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
    claim_choice = st.radio("", [agree_text, disagree_text], index=initial_idx, key="claim_radio")
    if claim_choice is None:
        st.session_state.claim = None
        st.info("Please choose your claim to continue.")
        st.stop()
    else:
        new_claim = "agree" if claim_choice == agree_text else "disagree"
        if st.session_state.prev_claim is not None and new_claim != st.session_state.prev_claim:
            st.session_state.claim = new_claim
            reset_after_claim_change(keep_text=True)
            st.info("Claim changed. Evidence and Reasoning need to be re-checked.")
        else:
            st.session_state.claim = new_claim
        st.session_state.prev_claim = new_claim
        st.success(f"Selected claim: **{st.session_state.claim.capitalize()}**")

    # Step 2 — Evidence
    st.divider(); st.subheader("2) Evidence")
    st.markdown('<div class="inst">Now present your evidence. Use the data from the table and chart to support your claim.</div>',
                unsafe_allow_html=True)
    st.text_area(
        "", key="evidence_text", height=150,
        placeholder=('Write your evidence here. Then click "Get feedback" to see suggestions and refine. '),
        disabled=st.session_state.evidence_ok,
    )
    c1, c2 = st.columns(2)
    with c1:
        ev_btn = st.button("Get feedback", key="ev_btn",
                           disabled=st.session_state.evidence_ok, use_container_width=True)
    with c2:
        if st.session_state.evidence_ok:
            st.button("Unlock Evidence to Edit", on_click=unlock_evidence,
                      use_container_width=True,
                      help="Editing evidence will also require re-checking your reasoning.")

    if ev_btn:
        try:
            result = gpt_eval_evidence(st.session_state.claim, st.session_state.evidence_text)
            st.session_state.evidence_ok = result["passed"]
            st.session_state.evidence_fb = result["feedback"]
            st.session_state.submitted = False
            st.session_state.evidence_history.append({
                "claim": st.session_state.claim,
                "text": st.session_state.evidence_text.strip(),
                "label": result["label"],
                "confidence": result["confidence"],
                "feedback": result["feedback"],
                "passed": result["passed"],
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error while checking evidence: {e}")

    if st.session_state.evidence_fb:
        show_feedback_bar(st.session_state.evidence_fb, st.session_state.evidence_ok)

    curr_claim = st.session_state.claim
    ev_records = [r for r in st.session_state.evidence_history if r.get("claim") == curr_claim]
    st.markdown("**Your Evidence Attempts & Feedback**")
    other_ev = len(st.session_state.evidence_history) - len(ev_records)
    if other_ev > 0:
        st.caption(f"{other_ev} attempt(s) from the other claim are hidden.")
    _render_evidence_history(ev_records, curr_claim.capitalize())

    if not st.session_state.evidence_ok:
        st.info("Refine your Evidence until it passes to unlock Reasoning.")
        st.stop()

    # Step 3 — Reasoning
    st.divider(); st.subheader("3) Reasoning")
    st.markdown('<div class="inst">Describe how your evidence supports your claim. Use what you know about predators, prey, and ecosystem balance to explain how your data supports your claim.</div>',
                unsafe_allow_html=True)
    st.text_area(
        "", key="reasoning_text", height=160,
        placeholder=('Write your reasoning here. Explain the mechanism that links your data to the claim. '
                     'Then click "Get feedback" to see suggestions and refine.'),
        disabled=st.session_state.reasoning_ok,
    )
    d1, d2 = st.columns(2)
    with d1:
        rs_btn = st.button("Get feedback", key="rs_btn",
                           disabled=st.session_state.reasoning_ok, use_container_width=True)
    with d2:
        if st.session_state.reasoning_ok:
            st.button("Unlock Reasoning to Edit", on_click=unlock_reasoning, use_container_width=True)

    if rs_btn:
        try:
            result = gpt_eval_reasoning(st.session_state.claim, st.session_state.reasoning_text)
            st.session_state.reasoning_ok = result["passed"]
            st.session_state.reasoning_fb = result["feedback"]
            st.session_state.submitted = False
            st.session_state.reasoning_history.append({
                "claim": st.session_state.claim,
                "text": st.session_state.reasoning_text.strip(),
                "label": result["label"],
                "confidence": result["confidence"],
                "feedback": result["feedback"],
                "passed": result["passed"],
                "evidence": st.session_state.evidence_text.strip(),  # evidence snapshot
                "ts": datetime.now().isoformat(timespec="seconds"),
            })
            st.rerun()
        except Exception as e:
            st.error(f"Error while checking reasoning: {e}")

    if st.session_state.reasoning_fb:
        show_feedback_bar(st.session_state.reasoning_fb, st.session_state.reasoning_ok)

    st.markdown("**Your Reasoning Attempts & Feedback**")
    rs_records = [r for r in st.session_state.reasoning_history if r.get("claim") == curr_claim]
    other_rs = len(st.session_state.reasoning_history) - len(rs_records)
    if other_rs > 0:
        st.caption(f"{other_rs} attempt(s) from the other claim are hidden.")
    _render_reasoning_history(rs_records, curr_claim.capitalize())

    # Final submit
    if st.session_state.evidence_ok and st.session_state.reasoning_ok:
        st.divider()
        if st.button("Submit", type="primary", use_container_width=True):
            st.session_state.submitted = True
            st.success("✅ Submitted! Your claim, evidence, and reasoning have been recorded.")
            st.balloons()
    if st.session_state.submitted:
        st.caption("Thanks! You can still read your feedback above.")
