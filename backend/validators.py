import re

REQUIRED_SECTIONS = ["FINDINGS", "IMPRESSION", "LABELS", "RECOMMENDATIONS"]

def extract_section(text: str, name: str) -> str:
    pat = rf"^\s*{name}:\s*(.*?)(?=^\s*(FINDINGS|IMPRESSION|LABELS|RECOMMENDATIONS):|\Z)"
    m = re.search(pat, text, flags=re.MULTILINE | re.DOTALL)
    return (m.group(1).strip() if m else "")

def parse_labels(labels_block: str):
    one_line = " ".join(labels_block.splitlines()).strip()
    parts = [p.strip() for p in one_line.split(",") if p.strip()]
    return parts if parts else None

def kg_present_absent(kg_json: dict | None):
    present, absent = set(), set()
    if not kg_json or "entities" not in kg_json:
        return present, absent
    for (text, label) in kg_json.get("entities", []):
        t = str(text).strip().lower()
        if not t:
            continue
        if "absent" in label:
            absent.add(t)
        elif "present" in label:
            present.add(t)
    return present, absent

def validate_report(report: str, kg_json: dict | None):
    errors = []

    # structure
    for sec in REQUIRED_SECTIONS:
        if not re.search(rf"^\s*{sec}:\s*", report, flags=re.MULTILINE):
            errors.append(f"Missing section: {sec}")

    # labels format
    labels_block = extract_section(report, "LABELS")
    if parse_labels(labels_block) is None:
        errors.append("LABELS must be a single comma-separated list (one line).")

    # KG contradiction heuristic
    findings = extract_section(report, "FINDINGS").lower()
    impression = extract_section(report, "IMPRESSION").lower()
    body = findings + " " + impression

    _, absent = kg_present_absent(kg_json)
    for ent in absent:
        if ent in body and not re.search(rf"(no|without|absence of)\s+{re.escape(ent)}", body):
            errors.append(f"KG says '{ent}' ABSENT but report implies it's present.")
            break

    return {"ok": len(errors) == 0, "errors": errors}
