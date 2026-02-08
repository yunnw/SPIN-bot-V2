# llm.py
# Combined evaluation: evidence + reasoning together
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

ArgLabel = {
    "1. Agree claim": 1,
    "2. Disagree claim": 2,
    "3. Supportive Data/Evidence": 3,
    "4. Non-supportive Data/Evidence": 4,
    "5. Supportive OR valid reasoning": 5,
    "6. Alternative OR invalid reasoning": 6,
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


# ========== File loaders ==========
@st.cache_data(show_spinner=False)
def load_task_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_resource(show_spinner=False)
def load_prompts(prompts_dir: str) -> Dict[str, str]:
    """
    Expect these files in prompts_dir (your screenshot):
      - System_prompt_for_evidence.txt
      - System_prompt_for_reasoning.txt
      - System_prompt_for_reasoning_pattern.txt
      - System_prompt_for_feedback_initial.txt
      - System_prompt_for_feedback_revision.txt
    """
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


def parse_single_evidence(highlight_str: str) -> Optional[Dict]:
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

    return {
        "type": it.get("Type", "").strip().lower(),
        "quote": it.get("Quote", "").strip(),
        "is_grounded_in_table": it.get("is_grounded_in_table", "").strip().lower(),  # yes/no/unclear
        "matches_table": it.get("matches_table", "").strip().lower(),                # yes/no/unclear
        "supports_claim_side": it.get("supports_claim_side", "").strip().lower(),    # yes/no/unclear
    }


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
        evidence_label = "Supportive Data/Evidence"
    elif "4" in arg_ids:
        evidence_label = "Non-supportive Data/Evidence"

    if "5" in arg_ids:
        reasoning_label = "Supportive OR valid reasoning"
    elif "6" in arg_ids:
        reasoning_label = "Alternative OR invalid reasoning"

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


def build_evidence_fewshot(cfg: dict, claim_side: str) -> str:
    ex = cfg["few_shot"]["evidence"][claim_side]
    sup = ex.get("supportive", [])
    nonsup = ex.get("non_supportive", [])
    lines = ["#### Supportive evidence/data:"]
    lines += [f'- "{s}"' for s in sup]
    lines += ["", "#### Non-supportive evidence/data:"]
    lines += [f'- "{s}"' for s in nonsup]
    return "\n".join(lines).strip()


def build_reasoning_fewshot(cfg: dict, claim_side: str) -> str:
    ex = cfg["few_shot"]["reasoning"][claim_side]
    valid = ex.get("valid", [])
    invalid = ex.get("invalid", [])
    noreasoning = ex.get("noreasoning", [])
    lines = ["#### Valid reasoning:"]
    lines += [f'- "{s}"' for s in valid]
    lines += ["", "#### Alternative reasoning:"]
    lines += [f'- "{s}"' for s in invalid]
    lines += ["", "#### No reasoning:"]
    lines += [f'- "{s}"' for s in noreasoning]
    return "\n".join(lines).strip()


def build_pattern_fewshot(cfg: dict) -> str:
    ex = cfg["few_shot"]["pattern"]["any"]
    rel = ex.get("relational", [])
    ce = ex.get("cause_effect", [])
    lines = ["#### Relational reasoning:"]
    lines += [f'- "{s}"' for s in rel]
    lines += ["", "#### Cause-effect reasoning:"]
    lines += [f'- "{s}"' for s in ce]
    return "\n".join(lines).strip()


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
    """
    Returns:
      - if check_highlights=True: (highlights_str, think_str)
      - else: (full_text, "")
    """
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


def evaluate_round(
    claim_choice: str,
    evidence_text: str,
    reasoning_text: str,
    chat_history: List[Dict],
    prompts_dir: str = "prompts/v4.2",
    task_yaml_path: str = "prompts/v4.2/tasks/rootworm.yaml",
    if_generate_feedback: bool = True,
) -> Dict:
    """
    Main entry called by app.py.
    - claim_choice: 'agree' | 'disagree'
    - chat_history: ONLY the history for THIS claim (app.py already filters)
      Each item ideally contains: student_resp, predicted_labels, evidence_info, feedback
    """
    if claim_choice not in ("agree", "disagree"):
        raise ValueError("claim_choice must be 'agree' or 'disagree'")

    prompts = load_prompts(prompts_dir)
    task_cfg = load_task_yaml(task_yaml_path)
    if not task_cfg:
        raise RuntimeError(f"Task yaml not found or empty: {task_yaml_path}")

    # Build a single student response string (include claim text for robustness)
    claim_text_for_prompt = task_cfg["claim_sides"][claim_choice]
    student_resp = (
        f"Claim:\n{claim_text_for_prompt}\n\n"
        f"Evidence:\n{(evidence_text or '').strip()}\n\n"
        f"Reasoning:\n{(reasoning_text or '').strip()}\n"
    ).strip()

    # Build system prompts (fill placeholders)
    evidence_prompt = prompts["evidence"].format(
        task_context=task_cfg["task_context"],
        claim_side_upper=claim_choice.upper(),
        claim_text=claim_text_for_prompt,
        fewshot_evidence_block=build_evidence_fewshot(task_cfg, claim_choice),
    )
    reasoning_prompt = prompts["reasoning"].format(
        task_context=task_cfg["task_context"],
        claim_side_upper=claim_choice.upper(),
        claim_text=claim_text_for_prompt,
        fewshot_reasoning_block=build_reasoning_fewshot(task_cfg, claim_choice),
    )
    pattern_prompt = prompts["reasoning_pattern"].format(
        task_context=task_cfg["task_context"],
        fewshot_pattern_block=build_pattern_fewshot(task_cfg),
        relational_reasoning_def=task_cfg.get("reasoning_pattern_definitions", {}).get("relational_reasoning_def", ""),
        cause_effect_reasoning_def=task_cfg.get("reasoning_pattern_definitions", {}).get("cause_effect_reasoning_def", ""),
    )

    # Parallel classify evidence + reasoning
    arg_pred_ids: set[str] = set()
    rea_pred_ids: set[str] = set()
    evidence_info: Optional[Dict] = None
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
                evidence_info = parse_single_evidence(hl_str)
            elif name == "reasoning":
                r_ids = labels_from_highlights(hl_str, {"5", "6"})
                if r_ids:
                    reasoning_found = True
                arg_pred_ids |= r_ids

    # If reasoning exists, classify reasoning pattern
    if reasoning_found:
        hl_str, _ = calling_azure_openai(student_resp, pattern_prompt, check_highlights=True)
        rea_pred_ids |= labels_from_highlights(hl_str, {"1", "2"})

    # Add claim id
    claim_id, claim_label_text = CLAIM_CHOICE_MAP[claim_choice]
    arg_pred_ids.add(claim_id)

    predicted_labels = generate_classification_from_pred(arg_pred_ids, rea_pred_ids)

    # Evidence info block used by feedback prompts
    if evidence_info is None:
        evidence_block = (
            "Evidence analysis (from the evidence evaluator):\n"
            "- Detected evidence: NONE\n"
        )
    else:
        evidence_block = (
            "Evidence analysis (from the evidence evaluator):\n"
            f'- Detected evidence quote: "{evidence_info["quote"]}"\n'
            f'- Type: {evidence_info["type"]}\n'
            f'- is_grounded_in_table: {evidence_info["is_grounded_in_table"]}\n'
            f'- matches_table: {evidence_info["matches_table"]}\n'
            f'- supports_claim_side: {evidence_info["supports_claim_side"]}\n'
        )

    feedback_text = ""
    if if_generate_feedback:
        if len(chat_history) == 0:
            feedback_prompt = prompts["feedback_initial"].format(
                task_context=task_cfg["task_context"],
                fewshot_feedback_block=task_cfg["few_shot"]["feedback"],
            )
            fb_query = (
                f"Student's stance on the claim: {claim_label_text}\n"
                f"Student's Response:\n{student_resp}\n\n"
                f"Classification Results:\n"
                f"Claim: {predicted_labels['claim']}\n"
                f"Evidence: {predicted_labels['evidence']}\n"
                f"Evidence Info: {evidence_block}\n"
                f"Reasoning: {predicted_labels['reasoning']}\n"
                f"Reasoning Pattern: {predicted_labels.get('reasoning_pattern', 'N/a')}"
            )
        else:
            first_round = chat_history[0]
            first_ev_info = first_round.get("evidence_info")

            if first_ev_info is None:
                first_evidence_block = (
                    "Evidence analysis (from the evidence evaluator):\n"
                    "- Detected evidence: NONE\n"
                )
            else:
                first_evidence_block = (
                    "Evidence analysis (from the evidence evaluator):\n"
                    f'- Detected evidence quote: "{first_ev_info["quote"]}"\n'
                    f'- Type: {first_ev_info["type"]}\n'
                    f'- is_grounded_in_table: {first_ev_info["is_grounded_in_table"]}\n'
                    f'- matches_table: {first_ev_info["matches_table"]}\n'
                    f'- supports_claim_side: {first_ev_info["supports_claim_side"]}\n'
                )

            feedback_prompt = prompts["feedback_revision"].format(
                task_context=task_cfg["task_context"],
                fewshot_feedback_block=task_cfg["few_shot"]["feedback"],
            )

            current_round_index = len(chat_history) + 1
            fb_query = (
                f"Task context:\n{task_cfg['task_context']}\n\n"
                f"CLAIM SIDE:\n{claim_label_text}\n\n"
                f"=== ORIGINAL RESPONSE (Round 1) ===\n"
                f"{first_round.get('student_resp','')}\n\n"
                f"Original classification:\n"
                f"Claim: {first_round['predicted_labels']['claim']}\n"
                f"Evidence: {first_round['predicted_labels']['evidence']}\n"
                f"Reasoning: {first_round['predicted_labels']['reasoning']}\n"
                f"Reasoning Pattern: {first_round['predicted_labels'].get('reasoning_pattern', 'N/a')}\n"
                f"{first_evidence_block}\n"
                f"Original feedback to the student:\n"
                f"{first_round.get('feedback','')}\n\n"
                f"=== CURRENT RESPONSE (Round {current_round_index}) ===\n"
                f"{student_resp}\n\n"
                f"Current classification:\n"
                f"Claim: {predicted_labels['claim']}\n"
                f"Evidence: {predicted_labels['evidence']}\n"
                f"Reasoning: {predicted_labels['reasoning']}\n"
                f"Reasoning Pattern: {predicted_labels.get('reasoning_pattern', 'N/a')}\n"
                f"{evidence_block}\n"
            )

        feedback_text, _ = calling_azure_openai(fb_query, feedback_prompt, check_highlights=False, temperature=0.8)
        feedback_text = feedback_text.strip()
    else:
        feedback_text = "[Feedback generation disabled]"

    return {
        "predicted_labels": predicted_labels,
        "evidence_info": evidence_info,
        "feedback": feedback_text,
        # for revision prompt usage in later rounds
        "student_resp": student_resp,
    }