# llm.py
# Combined evaluation: evidence + reasoning together (v4.3)
from __future__ import annotations

import json
import time
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
import streamlit as st

AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT = st.secrets["AZURE_DEPLOYMENT"]
AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION", "2025-01-01-preview")

# ============================================================
# Label maps  (v4.3 – matches interactive script)
# ============================================================
ArgLabel = {
    "1. Agree claim": 1,
    "2. Disagree claim": 2,
    "3. Supportive Evidence/Data": 3,
    "4. Non-supportive Evidence/Data": 4,
    "5. Valid reasoning": 5,
    "6. Alternative reasoning": 6,
}
ReaLabel = {
    "1. Relational reasoning": 1,
    "2. Cause-effect reasoning": 2,
}

CLAIM_CHOICE_MAP = {
    "agree": ("1", "Agree Claim"),
    "disagree": ("2", "Disagree Claim"),
}

ARG_TEXT2ID = {k.split(". ", 1)[-1].lower(): str(v) for k, v in ArgLabel.items()}
REA_TEXT2ID = {k.split(". ", 1)[-1].lower(): str(v) for k, v in ReaLabel.items()}


# ============================================================
# File loaders
# ============================================================
@st.cache_data(show_spinner=False)
def load_task_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_resource(show_spinner=False)
def load_prompts(prompts_dir: str) -> Dict[str, str]:
    files = {
        "evidence": f"{prompts_dir}/System_prompt_for_evidence.txt",
        "reasoning": f"{prompts_dir}/System_prompt_for_reasoning.txt",
        "reasoning_pattern": f"{prompts_dir}/System_prompt_for_reasoning_pattern.txt",
        "feedback_initial": f"{prompts_dir}/System_prompt_for_feedback_initial.txt",
        "feedback_revision": f"{prompts_dir}/System_prompt_for_feedback_revision.txt",
    }
    out = {}
    for k, p in files.items():
        with open(p, "r", encoding="utf-8") as f:
            out[k] = f.read().strip()
    return out


# ============================================================
# JSON helpers
# ============================================================
def quick_fix_json(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "[]"
    s = s.replace("}{", "},{")
    if not s.startswith("["):
        s = "[" + s
    if not s.endswith("]"):
        s = s + "]"
    return s


def _extract_json_array_fragment(s: str) -> str:
    if not s:
        return "[]"
    m = re.search(r"\[[\s\S]*\]", s)
    return m.group(0) if m else s


# ============================================================
# Label extraction
# ============================================================
def labels_from_highlights(highlight_str: str, valid_ids: set[str]) -> set[str]:
    ids: set[str] = set()
    if not highlight_str or highlight_str.strip() == "[]":
        return ids

    highlight_str = _extract_json_array_fragment(highlight_str)
    try:
        items = json.loads(highlight_str)
    except Exception:
        items = json.loads(quick_fix_json(highlight_str))

    for it in items:
        if isinstance(it, dict):
            if "id" in it and str(it["id"]) in valid_ids:
                ids.add(str(it["id"]))
            t = it.get("Type", "").strip().lower()
            if t in ARG_TEXT2ID and ARG_TEXT2ID[t] in valid_ids:
                ids.add(ARG_TEXT2ID[t])
            if t in REA_TEXT2ID and REA_TEXT2ID[t] in valid_ids:
                ids.add(REA_TEXT2ID[t])
    return ids


def generate_classification_from_pred(arg_ids: set[str], rea_ids: set[str]) -> Dict[str, str]:
    claim_label = "No claim"
    evidence_label = "No evidence"
    reasoning_label = "No reasoning"
    reasoning_pattern = "N/a"

    if "1" in arg_ids:
        claim_label = "Agree Claim"
    elif "2" in arg_ids:
        claim_label = "Disagree Claim"

    if "3" in arg_ids:
        evidence_label = "Supportive Evidence/Data"
    elif "4" in arg_ids:
        evidence_label = "Non-supportive Evidence/Data"

    if "5" in arg_ids:
        reasoning_label = "Valid reasoning"
    elif "6" in arg_ids:
        reasoning_label = "Alternative reasoning"

    if "1" in rea_ids:
        reasoning_pattern = "Relational"
    elif "2" in rea_ids:
        reasoning_pattern = "Cause-Effect"

    return {
        "claim": claim_label,
        "evidence": evidence_label,
        "reasoning": reasoning_label,
        "reasoning_pattern": reasoning_pattern,
    }


# ============================================================
# Unified parse / format helpers  (matching interactive v4.3)
# ============================================================
def parse_single_highlight(highlight_str: str, fields: list[str]) -> Optional[Dict]:
    """Unified parser – replaces old parse_single_evidence."""
    if not highlight_str or highlight_str.strip() == "[]":
        return None

    highlight_str = _extract_json_array_fragment(highlight_str)
    try:
        items = json.loads(highlight_str)
    except Exception:
        items = json.loads(quick_fix_json(highlight_str))

    if not items:
        return None

    it = items[0]
    if not isinstance(it, dict):
        return None

    result = {
        "type": it.get("Type", "").strip().lower(),
        "quote": it.get("Quote", "").strip(),
    }
    for f in fields:
        result[f] = it.get(f, "").strip().lower()
    return result


def format_evidence_block(info: Optional[Dict]) -> str:
    if info is None:
        return (
            "Evidence analysis (from the evidence evaluator):\n"
            "- Detected evidence: NONE\n"
        )
    return (
        "Evidence analysis (from the evidence evaluator):\n"
        f'- Detected evidence quote: "{info["quote"]}"\n'
        f'- Type: {info["type"]}\n'
        f'- is_grounded_in_table: {info["is_grounded_in_table"]}\n'
        f'- matches_table: {info["matches_table"]}\n'
        f'- supports_claim_side: {info["supports_claim_side"]}\n'
    )


def format_reasoning_block(info: Optional[Dict]) -> str:
    if info is None:
        return (
            "Reasoning analysis (from the reasoning evaluator):\n"
            "- Detected reasoning: NONE\n"
        )
    return (
        "Reasoning analysis (from the reasoning evaluator):\n"
        f'- Detected reasoning quote: "{info["quote"]}"\n'
        f'- Type: {info["type"]}\n'
        f'- is_reasoning: {info["is_reasoning"]}\n'
        f'- connects_evidence_to_claim: {info["connects_evidence_to_claim"]}\n'
        f'- is_scientifically_valid: {info["is_scientifically_valid"]}\n'
    )


def format_classification_summary(labels: Dict, ev_block: str, re_block: str) -> str:
    return (
        f"Claim: {labels['claim']}\n"
        f"Evidence: {labels['evidence']}\n"
        f"Reasoning: {labels['reasoning']}\n"
        f"Reasoning Pattern: {labels.get('reasoning_pattern', 'N/a')}\n"
        f"{ev_block}\n"
        f"{re_block}\n"
    )


def build_fewshot_section(cfg_dict: dict, sections: list[tuple[str, str]]) -> str:
    """Unified fewshot builder – replaces build_evidence_fewshot / build_reasoning_fewshot / build_pattern_fewshot."""
    lines = []
    for title, key in sections:
        lines.append(f"#### {title}:")
        lines += [f'- "{s}"' for s in cfg_dict.get(key, [])]
        lines.append("")
    return "\n".join(lines).strip()


def decide_feedback_focus(labels: Dict) -> str:
    """Priority: evidence → reasoning → reasoning_pattern."""
    if labels["evidence"] in ("No evidence", "Non-supportive Evidence/Data"):
        return "evidence"
    if labels["reasoning"] in ("No reasoning", "Alternative reasoning"):
        return "reasoning"
    return "reasoning_pattern"


# ============================================================
# Azure API
# ============================================================
def _call_azure_chat(system_prompt: str, user_query: str, temperature: float, timeout: int = 60) -> str:
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def calling_azure_openai(
    query: str,
    prompt: str,
    retry: int = 3,
    check_highlights: bool = True,
    temperature: float = 0.8,
) -> Tuple[str, str]:
    last_err: Optional[Exception] = None

    for attempt in range(retry + 1):
        try:
            content = _call_azure_chat(prompt, query, temperature=temperature, timeout=60)

            if not check_highlights:
                return content, ""

            if "HIGHLIGHTS=" not in content:
                if attempt == retry:
                    return "[]", "Error: Fail to extract anything!!!"
                time.sleep(1.5 ** attempt)
                continue

            think = content.split("</think>")[0] if "</think>" in content else content.split("HIGHLIGHTS=", 1)[0]
            hl = content.split("HIGHLIGHTS=", 1)[-1].strip()
            hl = _extract_json_array_fragment(hl)
            return hl, think

        except Exception as e:
            last_err = e
            time.sleep(1.5 ** attempt)

    return "[]", f"Exception occurred: {last_err}"


def run_component_prompt(name: str, student_resp: str, prompt_str: str) -> Tuple[str, str, str]:
    hl_str, think = calling_azure_openai(student_resp, prompt_str, check_highlights=True)
    return name, hl_str, think


# ============================================================
# Main entry  (called by app.py)
# ============================================================
def evaluate_round(
    claim_choice: str,
    evidence_text: str,
    reasoning_text: str,
    chat_history: List[Dict],
    prompts_dir: str = "prompts/v4.3",
    task_yaml_path: str = "prompts/v4.3/tasks/rootworm.yaml",
    if_generate_feedback: bool = True,
) -> Dict:
    """
    - claim_choice: 'agree' | 'disagree'
    - chat_history: history for THIS claim only, each item has:
        round_index, student_resp, predicted_labels, evidence_info,
        reasoning_info, feedback
    """
    if claim_choice not in ("agree", "disagree"):
        raise ValueError("claim_choice must be 'agree' or 'disagree'")

    prompts = load_prompts(prompts_dir)
    task_cfg = load_task_yaml(task_yaml_path)
    if not task_cfg:
        raise RuntimeError(f"Task yaml not found or empty: {task_yaml_path}")

    claim_text_for_prompt = task_cfg["claim_sides"][claim_choice]
    student_resp = (
        f"Claim:\n{claim_text_for_prompt}\n\n"
        f"Evidence:\n{(evidence_text or '').strip()}\n\n"
        f"Reasoning:\n{(reasoning_text or '').strip()}\n"
    ).strip()

    # ---- Build system prompts ----
    evidence_prompt = prompts["evidence"].format(
        task_context=task_cfg["task_context"],
        claim_side_upper=claim_choice.upper(),
        claim_text=claim_text_for_prompt,
        fewshot_evidence_block=build_fewshot_section(
            task_cfg["few_shot"]["evidence"][claim_choice],
            [("Supportive evidence/data", "supportive"),
             ("Non-supportive evidence/data", "non_supportive")],
        ),
    )
    reasoning_prompt = prompts["reasoning"].format(
        task_context=task_cfg["task_context"],
        claim_side_upper=claim_choice.upper(),
        claim_text=claim_text_for_prompt,
        fewshot_reasoning_block=build_fewshot_section(
            task_cfg["few_shot"]["reasoning"][claim_choice],
            [("Valid reasoning", "valid"),
             ("Alternative reasoning", "invalid"),
             ("No reasoning", "noreasoning")],
        ),
    )
    pattern_prompt = prompts["reasoning_pattern"].format(
        task_context=task_cfg["task_context"],
        fewshot_pattern_block=build_fewshot_section(
            task_cfg["few_shot"]["pattern"]["any"],
            [("Relational reasoning", "relational"),
             ("Cause-effect reasoning", "cause_effect")],
        ),
        relational_reasoning_def=task_cfg.get("reasoning_pattern_definitions", {}).get("relational_reasoning_def", ""),
        cause_effect_reasoning_def=task_cfg.get("reasoning_pattern_definitions", {}).get("cause_effect_reasoning_def", ""),
    )

    # ---- Parallel classify evidence + reasoning ----
    arg_pred_ids: set[str] = set()
    rea_pred_ids: set[str] = set()
    evidence_info: Optional[Dict] = None
    reasoning_info: Optional[Dict] = None
    reasoning_found = False

    active = {
        "evidence": evidence_prompt,
        "reasoning": reasoning_prompt,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_component_prompt, name, student_resp, pstr): name
            for name, pstr in active.items()
        }
        for fut in as_completed(futures):
            name, hl_str, _ = fut.result()
            if name == "evidence":
                arg_pred_ids |= labels_from_highlights(hl_str, {"3", "4"})
                evidence_info = parse_single_highlight(
                    hl_str,
                    ["is_grounded_in_table", "matches_table", "supports_claim_side"],
                )
            elif name == "reasoning":
                r_ids = labels_from_highlights(hl_str, {"5", "6"})
                reasoning_info = parse_single_highlight(
                    hl_str,
                    ["is_reasoning", "connects_evidence_to_claim", "is_scientifically_valid"],
                )
                if r_ids:
                    reasoning_found = True
                arg_pred_ids |= r_ids

    # ---- Reasoning pattern (only if reasoning found) ----
    if reasoning_found:
        hl_str, _ = calling_azure_openai(student_resp, pattern_prompt, check_highlights=True)
        rea_pred_ids |= labels_from_highlights(hl_str, {"1", "2"})

    # ---- Claim id ----
    claim_id, claim_label_text = CLAIM_CHOICE_MAP[claim_choice]
    arg_pred_ids.add(claim_id)

    predicted_labels = generate_classification_from_pred(arg_pred_ids, rea_pred_ids)

    # ---- Build feedback ----
    evidence_block = format_evidence_block(evidence_info)
    reasoning_block = format_reasoning_block(reasoning_info)
    focus = decide_feedback_focus(predicted_labels)

    feedback_text = ""
    if if_generate_feedback:
        if len(chat_history) == 0:
            # ---- Initial feedback ----
            feedback_prompt = prompts["feedback_initial"].format(
                task_context=task_cfg["task_context"],
                fewshot_feedback_block=task_cfg["few_shot"]["feedback"],
            )
            fb_query = (
                f"Student's stance on the claim: {claim_label_text}\n"
                f"Student's Response:\n{student_resp}\n\n"
                f"Classification Results:\n"
                + format_classification_summary(predicted_labels, evidence_block, reasoning_block)
                + f"\n** FEEDBACK FOCUS FOR THIS ROUND: {focus.upper()} **\n"
            )
        else:
            # ---- Revision feedback with full history ----
            feedback_prompt = prompts["feedback_revision"].format(
                task_context=task_cfg["task_context"],
                fewshot_feedback_block=task_cfg["few_shot"]["feedback"],
            )

            history_text = ""
            for r in chat_history:
                r_idx = r["round_index"]
                r_ev = format_evidence_block(r.get("evidence_info"))
                r_re = format_reasoning_block(r.get("reasoning_info"))
                history_text += (
                    f"=== Round {r_idx} ===\n"
                    f"Student response:\n{r.get('student_resp', '')}\n\n"
                    f"Classification:\n"
                    + format_classification_summary(r["predicted_labels"], r_ev, r_re)
                    + f"Feedback given:\n{r.get('feedback', '')}\n\n"
                )

            current_round_index = len(chat_history) + 1

            fb_query = (
                f"CLAIM SIDE: {claim_label_text}\n\n"
                f"=== RESPONSE HISTORY ===\n"
                f"{history_text}"
                f"=== CURRENT RESPONSE (Round {current_round_index}) ===\n"
                f"{student_resp}\n\n"
                f"Current classification:\n"
                + format_classification_summary(predicted_labels, evidence_block, reasoning_block)
                + f"\n** FEEDBACK FOCUS FOR THIS ROUND: {focus.upper()} **\n"
            )

        feedback_text, _ = calling_azure_openai(fb_query, feedback_prompt, check_highlights=False, temperature=0.8)
        feedback_text = feedback_text.strip()
    else:
        feedback_text = "[Feedback generation disabled]"

    return {
        "predicted_labels": predicted_labels,
        "evidence_info": evidence_info,
        "reasoning_info": reasoning_info,
        "feedback": feedback_text,
        "student_resp": student_resp,
    }