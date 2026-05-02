from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


DISEASES = [
    "Cardiomegaly",
    "Pleural Effusion",
    "Edema",
    "Pneumothorax",
    "Infiltrate",
    "Consolidation",
    "Lung Opacity",
    "Nodule",
    "Atelectasis",
    "Fracture",
]

KEYWORD_MAP = {
    "Cardiomegaly": [
        "cardiomegaly", "cardiac enlargement", "heart is enlarged",
        "heart size is enlarged", "enlarged heart", "heart is large",
        "enlargement of the cardiac silhouette", "enlarged cardiac silhouette",
        "moderate-to-marked enlargement of the cardiac silhouette",
        "heart is moderately enlarged", "heart is mildly enlarged",
        "heart is severely enlarged", "cardiac silhouette is enlarged",
        "cardiomediastinal silhouette is enlarged",
        "stable enlargement of the heart", "stable enlarged heart",
        "heart size enlarged", "mild cardiomegaly",
        "moderate cardiomegaly", "stable cardiomegaly",
        "mildly enlarged heart", "moderately enlarged heart",
    ],
    "Pleural Effusion": [
        "pleural effusion", "pleural effusions", "effusions",
        "pleural fluid", "posterior pleural effusion",
        "costophrenic blunting", "costophrenic angle blunting",
        "costophrenic sulcus blunting", "costophrenic recess blunting",
        "blunting of the costophrenic", "blunting of bilateral costophrenic",
        "blunting of the bilateral costophrenic", "blunted costophrenic",
        "blunted posterior costophrenic",
    ],
    "Edema": [
        "edema", "pulmonary edema", "vascular congestion",
        "pulmonary congestion", "fluid overload", "interstitial edema",
        "central vascular congestion", "pulmonary xxxx are engorged",
    ],
    "Pneumothorax": [
        "pneumothorax", "pneumothoraces", "pleural air",
        "pleural air collection",
    ],
    "Infiltrate": [
        "infiltrate", "infiltrates", "airspace disease",
        "airspace opacity", "air space opacity", "air space opacities",
        "airspace consolidation",
    ],
    "Consolidation": [
        "consolidation", "consolidations", "focal consolidation",
        "lobar consolidation", "airspace process", "airspace processes",
        "airspace infiltrate", "focal airspace disease",
        "alveolar consolidation",
    ],
    "Lung Opacity": [
        "opacity", "opacities", "opacification", "haziness",
        "hazy opacity", "airspace opacity",
    ],
    "Nodule": [
        "nodule", "nodules", "mass", "calcified nodule",
        "pulmonary nodule", "lung nodule", "solitary pulmonary nodule",
        "pulmonary mass", "lung mass", "coin lesion",
    ],
    "Atelectasis": [
        "atelectasis", "atelectatic", "volume loss",
    ],
    "Fracture": [
        "fracture", "fractures", "rib fracture", "compression fracture",
        "wedge fracture", "wedge-shaped fracture", "wedge deformity",
        "healing deformity", "rib deformity", "healed rib fracture",
        "remote fracture", "minimally displaced fracture", "displaced fracture",
    ],
}

NEGATION_WINDOW = 120
NEGATION_PHRASES = [
    "no ", "no evidence", "without ", "negative for", "free of",
    "clear of", "absent", "not ", "denies ", "ruled out",
    "resolution of", "resolved", "removed", "no definite",
    "no definitive", "no visible", "no acute", "no focal",
    "no large", "no obvious", "no significant", "no suspicious",
    "no displaced", "nondisplaced",
]
UNCERTAINTY_PHRASES = [
    "possible", "possibly", "may represent", "may be", "may reflect",
    "suggesting", "suggestive of", "questionable", "cannot exclude",
    "versus", "vs.", "probable", "probably", "likely", "suspicious for",
    "favored to represent", "could represent", "may indicate",
    "small effusion versus", "scar versus", "thickening versus",
    "evaluation for", "limited evaluation", "is limited", "limited by",
    "not well evaluated", "difficult to evaluate", "cannot be assessed",
]
IGNORE_CONTEXT = {
    "Nodule": [
        "granuloma", "granulomas", "granulomatous", "calcified granuloma",
        "calcified granulomas", "calcified hilar", "calcified mediastinal",
    ],
    "Fracture": [
        "remote", "old", "healed", "healing", "chronic", "stable",
        "age-indeterminate", "deformity", "degenerative",
    ],
}


@dataclass
class DiseaseEvidence:
    keyword: str
    context_before: str
    context_after: str


def _keyword_context(report: str, pos: int) -> str:
    context_start = max(0, pos - NEGATION_WINDOW)
    context = report[context_start:pos]
    last_boundary = max(context.rfind("."), context.rfind("!"), context.rfind("?"))
    if last_boundary >= 0:
        context = context[last_boundary + 1:]
    return context


def _normalize_evidence_item(keyword: str, context_before: str, context_after: str) -> Dict[str, str]:
    return {
        "keyword": keyword,
        "context_before": context_before.strip(),
        "context_after": context_after.strip(),
    }


def collect_keyword_evidence(report: str, disease: str, keywords: Iterable[str]) -> Dict:
    evidence = {
        "positive_keywords": [],
        "uncertain_keywords": [],
        "negated_keywords": [],
        "ignored_keywords": [],
    }
    for keyword in keywords:
        if keyword not in report:
            continue
        start = 0
        while True:
            pos = report.find(keyword, start)
            if pos == -1:
                break
            context_before = _keyword_context(report, pos)
            context_after = report[pos: min(len(report), pos + NEGATION_WINDOW)]
            evidence_item = _normalize_evidence_item(keyword, context_before, context_after)

            negated = any(neg in context_before for neg in NEGATION_PHRASES)
            ignored = any(
                ignore in context_before or ignore in context_after
                for ignore in IGNORE_CONTEXT.get(disease, [])
            )
            uncertain = any(
                phrase in context_before or phrase in context_after
                for phrase in UNCERTAINTY_PHRASES
            )

            if negated:
                evidence["negated_keywords"].append(evidence_item)
            elif ignored:
                evidence["ignored_keywords"].append(evidence_item)
            elif uncertain:
                evidence["uncertain_keywords"].append(evidence_item)
            else:
                evidence["positive_keywords"].append(evidence_item)
            start = pos + len(keyword)
    return evidence


def certainty_labels_from_report(report: str) -> Dict[str, str]:
    normalized_report = (report or "").lower()
    statuses: Dict[str, str] = {}
    for disease in DISEASES:
        evidence = collect_keyword_evidence(normalized_report, disease, KEYWORD_MAP[disease])
        if evidence["positive_keywords"]:
            statuses[disease] = "present"
        elif evidence["uncertain_keywords"]:
            statuses[disease] = "uncertain"
        elif evidence["negated_keywords"]:
            statuses[disease] = "absent"
        else:
            statuses[disease] = "not_mentioned"
    return statuses


def label_audit_from_report(report: str) -> Tuple[np.ndarray, Dict]:
    normalized_report = (report or "").lower()
    labels = np.zeros(len(DISEASES), dtype=np.float32)
    audit = {}
    certainty = certainty_labels_from_report(report)
    for idx, disease in enumerate(DISEASES):
        evidence = collect_keyword_evidence(normalized_report, disease, KEYWORD_MAP[disease])
        if evidence["positive_keywords"] or evidence["uncertain_keywords"]:
            labels[idx] = 1.0
        audit[disease] = {
            "label": int(labels[idx]),
            "certainty": certainty[disease],
            **evidence,
        }
    return labels, audit


def labels_from_report(report: str) -> np.ndarray:
    labels, _ = label_audit_from_report(report)
    return labels


def definite_positive_vector(report: str) -> np.ndarray:
    statuses = certainty_labels_from_report(report)
    return np.asarray([1.0 if statuses[disease] == "present" else 0.0 for disease in DISEASES], dtype=np.float32)


def uncertain_positive_vector(report: str) -> np.ndarray:
    statuses = certainty_labels_from_report(report)
    return np.asarray(
        [1.0 if statuses[disease] in {"present", "uncertain"} else 0.0 for disease in DISEASES],
        dtype=np.float32,
    )


def case_profile(report: str) -> Dict[str, object]:
    definite = definite_positive_vector(report)
    support = uncertain_positive_vector(report)
    certainty = certainty_labels_from_report(report)
    return {
        "certainty": certainty,
        "definite_vector": definite,
        "support_vector": support,
        "definite_positive_count": int(definite.sum()),
        "support_positive_count": int(support.sum()),
        "rare_positive_count": int(sum(
            certainty[disease] == "present"
            for disease in ("Pleural Effusion", "Edema", "Pneumothorax", "Consolidation", "Nodule", "Fracture")
        )),
    }
