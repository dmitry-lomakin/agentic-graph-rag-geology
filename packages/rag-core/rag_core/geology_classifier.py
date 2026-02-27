"""Keyword-based geology subdomain classifier.

Classifies text chunks into one of 8 geology subdomains based on
weighted keyword matching in Russian and English.
"""

from __future__ import annotations

# Each subdomain maps to a list of (keyword, weight) tuples.
# Keywords are lowercased for matching. Higher weight = stronger signal.
SUBDOMAIN_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "geostatistics": [
        ("кригинг", 3), ("kriging", 3), ("вариограмма", 3), ("variogram", 3),
        ("nugget", 2), ("наггет", 2), ("sill", 1.5), ("порог", 1),
        ("idw", 2), ("обратные расстояния", 2), ("inverse distance", 2),
        ("геостатистик", 2), ("geostatistic", 2), ("семивариограмма", 2),
        ("semivariogram", 2), ("anisotrop", 1.5), ("анизотроп", 1.5),
        ("search ellips", 1.5), ("эллипс поиска", 1.5),
    ],
    "block_modeling": [
        ("блочная модель", 3), ("block model", 3), ("субблок", 2),
        ("sub-block", 2), ("subblock", 2), ("block size", 2),
        ("размер блока", 2), ("grade estimation", 2),
        ("оценка содержан", 2), ("nnr", 1.5), ("интерполяция", 1.5),
        ("interpolat", 1.5), ("parent block", 1.5), ("parent cell", 1.5),
    ],
    "reserve_calc": [
        ("подсчёт запасов", 3), ("подсчет запасов", 3),
        ("reserve estimation", 3), ("resource estimation", 3),
        ("кондиции", 2.5), ("cut-off", 2.5), ("cutoff", 2),
        ("бортовое содержан", 2.5), ("категори", 1.5),
        ("с1", 1), ("с2", 1), ("measured", 1), ("indicated", 1),
        ("inferred", 1), ("минимальное промышленное", 2),
    ],
    "mine_planning": [
        ("оптимизация карьера", 3), ("pit optimization", 3),
        ("pit optimisation", 3), ("mine planning", 3),
        ("планирование горных работ", 3), ("lerchs-grossmann", 2.5),
        ("lerch", 2), ("scheduling", 2), ("календарное планирование", 2.5),
        ("бвр", 2), ("drill and blast", 2), ("буровзрывн", 2),
        ("whittle", 2), ("deswik", 2), ("open pit", 1.5),
        ("карьер", 1), ("underground", 1.5), ("подземн", 1),
    ],
    "drillhole_data": [
        ("скважин", 2), ("drill hole", 2.5), ("drillhole", 2.5),
        ("collar", 2), ("survey", 1.5), ("assay", 2.5),
        ("опробование", 2.5), ("литология", 2), ("lithology", 2),
        ("керн", 1.5), ("core", 1), ("инклинометр", 2),
        ("downhole", 2), ("забой", 1.5), ("проба", 1.5),
    ],
    "wireframe_modeling": [
        ("каркасная модель", 3), ("wireframe", 3),
        ("триангуляция", 2.5), ("triangulat", 2.5),
        ("solid model", 2.5), ("поверхность", 1.5), ("surface", 1),
        ("implicit model", 2.5), ("имплицитн", 2.5),
        ("leapfrog", 2), ("рудное тело", 2), ("ore body", 2),
    ],
    "regulatory": [
        ("гкз", 3), ("государственная экспертиза", 3),
        ("закон о недрах", 3), ("гост", 2), ("снип", 1.5),
        ("методические рекомендации", 2), ("приказ мпр", 2.5),
        ("классификация запасов", 2.5), ("форма 5-гр", 2.5),
        ("тэо кондиций", 2.5), ("роснедра", 2),
    ],
    "reporting": [
        ("отчёт", 1.5), ("отчет", 1.5), ("report", 1),
        ("jorc", 2.5), ("ni 43-101", 2.5), ("43-101", 2.5),
        ("competent person", 2), ("компетентное лицо", 2),
        ("public report", 2), ("техническ", 1),
    ],
}


def classify_subdomain(text: str) -> str:
    """Classify a chunk into a geology subdomain based on keyword matching.

    Returns the highest-scoring subdomain, or "general" if no keywords match.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for subdomain, keywords in SUBDOMAIN_KEYWORDS.items():
        score = 0.0
        for keyword, weight in keywords:
            if keyword in text_lower:
                score += weight
        if score > 0:
            scores[subdomain] = score

    if not scores:
        return "general"

    return max(scores, key=scores.get)  # type: ignore[arg-type]
