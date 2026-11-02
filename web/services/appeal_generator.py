"""
SafeLens Automated Appeal Draft Generator.

Drafts formal, legally sound YouTube Community Guidelines appeal letters
based on multimodal evidence, specific policy clause citations (e.g., YT-HATE-02),
Educational/Documentary/Scientific/Artistic (EDSA) exemptions, and actionable remediations.
"""

import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _format_time_range(start: Any, end: Any) -> str:
    """Format start and end timestamps into MM:SS - MM:SS format."""
    def _to_sec(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            parts = val.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            try:
                return float(val)
            except ValueError:
                return 0.0
        return 0.0

    s_sec = _to_sec(start)
    e_sec = _to_sec(end)
    s_min, s_rem = divmod(int(s_sec), 60)
    e_min, e_rem = divmod(int(e_sec), 60)
    return f"{s_min:02d}:{s_rem:02d} - {e_min:02d}:{e_rem:02d}"


def _build_rule_based_appeal(
    video_metadata: Dict[str, Any],
    flagged_clauses: List[Dict[str, Any]],
    edsa_events: List[Dict[str, Any]],
    remediations: List[Dict[str, Any]],
    events_summary: List[Dict[str, Any]],
) -> str:
    """Generate structured high-grade appeal letter without needing LLM."""
    title = video_metadata.get("title") or video_metadata.get("id") or "Submitted Video"
    video_id = video_metadata.get("id", "N/A")

    has_edsa = len(edsa_events) > 0
    clauses_cited_str = ", ".join(list({c.get("clause_id", "") for c in flagged_clauses if c.get("clause_id")})) or "General Community Guidelines"

    sections = [
        "# Formal Appeal for YouTube Community Guidelines Enforcement",
        "",
        f"**Date:** Request for Expedited Review  ",
        f"**Video Title:** {title}  ",
        f"**Video ID:** `{video_id}`  ",
        f"**Policy Clauses at Issue:** {clauses_cited_str}  ",
        "",
        "---",
        "",
        "### 1. Statement of Appeal & Request for Human Review",
        "",
        "Dear YouTube Trust & Safety Review Team,",
        "",
        "I am submitting this formal appeal to respectfully request human review and reconsideration of the enforcement action taken against the video referenced above. Upon comprehensive multimodal review of the flagged segments, I believe the automated detection system flagged this content out of context, without taking into account its documentary and educational framing under YouTube's official **Educational, Documentary, Scientific, or Artistic (EDSA)** guidelines.",
        "",
        "### 2. Itemized Breakdown of Flagged Segments",
        "",
    ]

    for idx, ev in enumerate(events_summary, 1):
        ts = ev.get("time_range", "Segment")
        cats = ", ".join(ev.get("categories", [])) or "Sensitive Content"
        expl = ev.get("explanation", "")
        clauses = ev.get("cited_clauses", [])
        edsa_note = ev.get("edsa_rationale", "")

        clause_names = [f"`{c.get('clause_id')}` ({c.get('clause_name')})" for c in clauses if c.get("clause_id")]
        clauses_txt = "; ".join(clause_names) if clause_names else "Policy review flag"

        sections.extend([
            f"#### Incident #{idx}: Timestamp [{ts}]",
            f"- **Enforcement Category:** {cats}",
            f"- **Cited Policy Clause:** {clauses_txt}",
            f"- **Contextual Overview:** {expl}",
        ])
        if edsa_note:
            sections.append(f"- **EDSA Policy Consideration:** {edsa_note}")
        sections.append("")

    sections.extend([
        "### 3. Substantive Legal & Contextual Defense (EDSA Framework)",
        "",
        "YouTube's official Community Guidelines explicitly establish that sensitive, historical, or graphic material is permissible when embedded in legitimate educational, documentary, scientific, or artistic contexts:",
        "",
    ])

    if has_edsa:
        sections.extend([
            "> **EDSA Protection Basis:**",
            "> The flagged material is presented strictly for documentary analysis and historical/educational edification. There is zero intent to incite violence, promote designated hate groups, celebrate abuse, or encourage unlawful conduct. The narrative tone remains objective and condemnatory toward harmful behavior.",
            "",
        ])
    else:
        sections.extend([
            "> **Absence of Harmful Intent:**",
            "> The visual or audio elements detected by automated filters occurred in an informational, non-promotional setting. At no point does the creator advocate for, endorse, or promote any violative acts.",
            "",
        ])

    if remediations:
        sections.extend([
            "### 4. Proactive Remediation & Creator Commitment",
            "",
            "To resolve any residual ambiguity and demonstrate good-faith cooperation with YouTube policies, I am fully prepared to apply the following creator-side remediations immediately:",
            "",
        ])
        for rem in remediations:
            action = rem.get("action", "remediation").upper()
            inst = rem.get("instruction", "Modify flagged portion")
            sections.append(f"- **[{action}]**: {inst}")
        sections.append("")

    sections.extend([
        "### 5. Conclusion & Relief Requested",
        "",
        "In light of the documentary nature of the video, the absence of incitement or malice, and YouTube's established EDSA policy precedents, I respectfully request that:",
        "1. A human specialist review the flagged timestamps within the full narrative context of the video.",
        "2. The Community Guidelines strike or restriction be removed, or alternatively, an appropriate age-gate (18+) or content advisory disclaimer be applied in lieu of a penalty.",
        "",
        "Thank you for your time, diligence, and continued support of responsible creators on YouTube.",
        "",
        "Sincerely,  ",
        "**Content Creator / SafeLens Compliance Review**",
    ])

    return "\n".join(sections)


def generate_appeal_letter(
    video_metadata: Dict[str, Any],
    harmful_events: List[Dict[str, Any]],
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate an appeal letter defending against flagged Community Guideline violations.

    Args:
        video_metadata: Video details (id, title, duration, etc.)
        harmful_events: List of detected harmful event dictionaries (from DB or safety report)
        llm: Optional SafetyLLM instance for customized neural drafting

    Returns:
        Dict containing appeal letter markdown, cited clauses, EDSA basis, and recommendations.
    """
    if not harmful_events:
        return {
            "video_id": video_metadata.get("id", ""),
            "title": video_metadata.get("title", ""),
            "appeal_letter_markdown": "# No Violations Detected\n\nThis video has no recorded policy violations; an appeal letter is not required.",
            "flagged_clauses": [],
            "edsa_claim_applicable": False,
            "edsa_summary": "No harmful events found.",
            "proposed_remediations": [],
            "appeal_strength": "N/A",
        }

    # Extract structured violation info
    all_cited_clauses: List[Dict[str, Any]] = []
    edsa_events: List[Dict[str, Any]] = []
    remediations: List[Dict[str, Any]] = []
    events_summary: List[Dict[str, Any]] = []

    for ev in harmful_events:
        # Normalize event attributes whether from DB row or report JSON
        start = ev.get("start_time", ev.get("segment_start", 0))
        end = ev.get("end_time", ev.get("segment_end", 0))
        time_range = _format_time_range(start, end)

        data = ev.get("analysis_data") or ev

        # Categories
        cats = data.get("categories", [])
        if isinstance(cats, str):
            try:
                cats = json.loads(cats)
            except Exception:
                cats = [cats]

        # Cited policies
        cited = data.get("cited_policy_clauses") or data.get("cited_policies") or []
        if isinstance(cited, str):
            try:
                cited = json.loads(cited)
            except Exception:
                cited = []

        if isinstance(cited, list):
            for c in cited:
                if isinstance(c, dict) and c.get("clause_id"):
                    all_cited_clauses.append(c)

        # EDSA
        is_edsa = bool(data.get("is_edsa_exempt", False))
        edsa_rat = data.get("edsa_rationale", "")
        if is_edsa or edsa_rat:
            edsa_events.append({"time_range": time_range, "rationale": edsa_rat})

        # Remediations / suggested edits
        edits = data.get("suggested_edits", [])
        if isinstance(edits, str):
            try:
                edits = json.loads(edits)
            except Exception:
                edits = []

        if isinstance(edits, list):
            for e in edits:
                if isinstance(e, dict):
                    remediations.append(e)

        events_summary.append({
            "time_range": time_range,
            "categories": cats,
            "explanation": data.get("explanation", ""),
            "cited_clauses": cited if isinstance(cited, list) else [],
            "edsa_rationale": edsa_rat,
        })

    has_edsa = len(edsa_events) > 0

    # Build prompt for LLM if available
    letter_markdown = None
    if llm is not None:
        try:
            incidents_text = "\n".join([
                f"- Timestamp [{ev['time_range']}]: Violations: {', '.join(ev['categories'])}. "
                f"Clauses: {', '.join([c.get('clause_id', '') for c in ev['cited_clauses']])}. "
                f"Context: {ev['explanation']}. EDSA Note: {ev['edsa_rationale']}"
                for ev in events_summary
            ])

            prompt = f"""You are a senior YouTube content policy attorney and creator defense advocate.
A YouTube creator's video was flagged for Community Guideline violations.
Generate a formal, compelling, legally sound appeal letter to YouTube Trust & Safety requesting human review.

VIDEO METADATA:
- Title: {video_metadata.get('title', 'Video Submission')}
- ID: {video_metadata.get('id', 'N/A')}

FLAGGED INCIDENTS:
{incidents_text}

APPEAL REQUIREMENTS:
1. Formal header with Video Title, ID, and cited policy clauses.
2. Request immediate human review under YouTube's Educational, Documentary, Scientific, or Artistic (EDSA) policy framework.
3. Defend each timestamp: explain documentary context, non-incitement, and absence of malicious intent.
4. Offer proactive remediation (e.g. voluntary blur/trim or age restriction 18+).
5. Professional, respectful closing ready to paste directly into YouTube Studio.

Provide the complete appeal in markdown format."""

            resp = llm.invoke(prompt, max_tokens=1200, temperature=0.3)
            if isinstance(resp, dict):
                content = resp.get("response") or resp.get("text") or resp.get("content")
                if isinstance(content, str) and len(content.strip()) > 100:
                    letter_markdown = content.strip()
            elif isinstance(resp, str) and len(resp.strip()) > 100:
                letter_markdown = resp.strip()
        except Exception as e:
            logger.warning(f"LLM appeal generation error: {e}. Falling back to structured appeal template.")

    # Fallback to rule-based structured appeal
    if not letter_markdown:
        letter_markdown = _build_rule_based_appeal(
            video_metadata=video_metadata,
            flagged_clauses=all_cited_clauses,
            edsa_events=edsa_events,
            remediations=remediations,
            events_summary=events_summary,
        )

    return {
        "video_id": video_metadata.get("id", ""),
        "title": video_metadata.get("title", ""),
        "appeal_letter_markdown": letter_markdown,
        "flagged_clauses": all_cited_clauses,
        "edsa_claim_applicable": has_edsa,
        "edsa_summary": f"{len(edsa_events)} segment(s) qualify under EDSA provisions" if has_edsa else "General context defense",
        "proposed_remediations": remediations,
        "appeal_strength": "High (Strong EDSA Documentary Basis)" if has_edsa else "Moderate (Contextual / Non-Promotion Basis)",
    }
