# llm.py  
# (evidence + reasoning only)
from __future__ import annotations
import json, re, time, os
from typing import Dict, Optional
import requests
import yaml
import streamlit as st


AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT = st.secrets["AZURE_DEPLOYMENT"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]

EVIDENCE_LABELS = {"supportive", "non_supportive"}
REASONING_LABELS = {"valid", "alternative"}

# ---------------- Prompts ----------------
def load_prompts(path: str = "prompts/v3.0.yml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return conf or {}

PROMPTS = load_prompts()

# ---------------- Low-level Chat ----------------
def _azure_chat(messages, temperature=0.3, timeout=60, max_retries=3, retry_backoff=1.5) -> str:
    if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
        raise RuntimeError(
            "Azure OpenAI config missing. Set env vars or Streamlit secrets: "
            "AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT."
        )
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    body = {
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"AOAI transient {resp.status_code}: {resp.text[:300]}")
                time.sleep(retry_backoff ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            time.sleep(retry_backoff ** attempt)
    raise last_err

# ---------------- JSON helpers ----------------
def _strip_code_fences(s: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.I | re.M)

def _json_only(s: str) -> dict:
    s = _strip_code_fences(s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)

# ---------------- Prompt inject ----------------
def _inject_vars_once(text: str) -> str:
    if not isinstance(text, str):
        return text
    common_desc = PROMPTS.get("common_desc", "")
    feedback_style = PROMPTS.get("feedback_style", "")
    evidence_common = PROMPTS.get("components", {}).get("evidence_common", "")
    reasoning_common = PROMPTS.get("components", {}).get("reasoning_common", "")
    return (text
            .replace("{{common_desc}}", common_desc)
            .replace("{{feedback_style}}", feedback_style)
            .replace("{{evidence_common}}", evidence_common)
            .replace("{{reasoning_common}}", reasoning_common))

def _inject_vars(text: str, passes: int = 3) -> str:
    if not isinstance(text, str):
        return text
    out = text
    for _ in range(passes):
        new_out = _inject_vars_once(out)
        if new_out == out:
            break
        out = new_out
    return out

def _get_prompt_component(name: str, claim_side: Optional[str] = None) -> str:
    """name ∈ {'evidence','reasoning'}"""
    comp = PROMPTS.get("components", {}).get(name, {})
    if isinstance(comp, dict):
        if claim_side and claim_side in comp:
            return _inject_vars(comp[claim_side])
        if "system" in comp:
            return _inject_vars(comp["system"])
    top = PROMPTS.get(name)
    if isinstance(top, str):
        return _inject_vars(top)
    raise KeyError(f"Prompt not found for component={name}, claim_side={claim_side}")

# ---------------- Public API ----------------
def _ask_component(prompt_system: str, user_content: str, temperature=0.3) -> Dict:
    content = _azure_chat(
        messages=[{"role": "system", "content": prompt_system},
                  {"role": "user", "content": user_content}],
        temperature=temperature
    )
    obj = _json_only(content)
    obj["label"] = str(obj.get("label", "")).strip().lower()
    obj["step_feedback"] = str(obj.get("step_feedback", "")).strip()
    try:
        obj["confidence"] = float(obj.get("confidence", 0))
    except Exception:
        obj["confidence"] = 0.0
    return obj

def step_feedback(component: str, claim_side: Optional[str], student_text: str, evidence_text: str = "") -> Dict:
    sys_prompt = _get_prompt_component(component, claim_side)
    user_tpl = PROMPTS.get("user_templates", {}).get(component, "{text}")

    if component == "reasoning":
        user_payload = user_tpl.format(
            reasoning=(student_text or "").strip(),
            evidence=(evidence_text or "").strip(),
            claim_side=claim_side or "",
        )
    else:
        user_payload = user_tpl.format(
            text=(student_text or "").strip(),
            claim_side=claim_side or "",
        )
    return _ask_component(sys_prompt, user_payload)

if __name__ == "__main__":
    print("llm.py ready (pattern removed).")
