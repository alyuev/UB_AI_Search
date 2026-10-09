"""Build the translation-quality benchmark set from REAL paragraphs of Paper 0 (the
Foreword) of urantia_bilingual_rag.json, with HAND-WRITTEN RU variants:

  * corrupted   - known, deliberate substantive errors (omission, addition, distortion,
                  wrong term, negation flip, wrong number, partial omission) -> must be flagged
  * paraphrase  - legitimate transformations (synonyms, reordering, voice change,
                  restructuring) that preserve meaning -> must NOT be flagged
  * unchanged   - the official RU text as-is -> must NOT be flagged (substantially)

Hand-written (not string-substitution) so every corruption is guaranteed to be real and
every paraphrase guaranteed to be meaning-preserving. Paper 0 only: small slice by design.

Run: python bench/translation/build_dataset.py   ->  bench/translation/dataset.json
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_JSON = os.path.join(ROOT, "urantia_bilingual_rag.json")
OUT_JSON = os.path.join(os.path.dirname(__file__), "dataset.json")

# (paragraph id, case type, expect_flagged, RU text or None for "official RU unchanged", note)
CASES = [
    # ---- corrupted: must be flagged ------------------------------------------------
    ("0:1.1", "omission", True,
     "Во вселенной вселенных представлены феномены деятельности божества на разнообразных "
     "уровнях космических реальностей, значений разума и ценностей духа.",
     "last clause 'but all of these ministrations ... are divinely co-ordinated' dropped"),
    ("0:1.17", "addition", True,
     "Божественность постигается созданиями как истина, красота и благость; коррелируется в "
     "личности как любовь, милосердие и служение; раскрывается на безличностных уровнях как "
     "правосудие, могущество и всевластие. Эти качества доступны только высшим расам.",
     "invented sentence about 'higher races' appended"),
    ("0:3.9", "distortion", True,
     "Как Первый Источник и Центр, Бог безусловно первичен по отношению ко всеобъемлющей "
     "реальности. Первый Источник и Центр бесконечен, равно как и вечен, и поэтому ограничен "
     "или обусловлен только законами материи.",
     "'limited only by volition' -> 'only by the laws of matter'"),
    ("0:1.14", "wrong_term", True,
     "Божество может быть экзистенциальным, как в Вечном Сыне; эмпирическим, как в Верховном "
     "Существе; ассоциативным, как в Боге-Семичастном; неделимым, как в Земной Троице.",
     "'Paradise Trinity' -> 'Earthly Trinity'"),
    ("0:6.6", "negation_flip", True,
     "2. Возникающая энергия охватывает все виды энергии, которые не реагируют на гравитацию "
     "Рая, но уже реагируют на локальную, или линейную, гравитацию. Это доэлектронный уровень "
     "энергии-вещества.",
     "both negations flipped"),
    ("0:3.14", "wrong_number", True,
     "2. Силы разума сходятся в Бесконечном Духе; дифференциальный и дивергентный космический "
     "разум – в Двенадцати Главных Духах; фактуализирующийся разум Верховного как "
     "пространственно-временно́й опыт – в Мажестоне.",
     "'Seven Master Spirits' -> 'Twelve'"),
    ("0:2.2", "partial_omission", True,
     "Космическое сознание подразумевает признание Первопричины. Бог – Всеобщий Отец – "
     "действует на трех уровнях личностного Божества, которые характеризуются суббесконечной "
     "ценностью и относительным выражением божественности:",
     "'the one and only uncaused reality' dropped"),
    # ---- acceptable paraphrases: must NOT be flagged ---------------------------------
    ("0:1.2", "paraphrase", False,
     "БОЖЕСТВО способно персонализироваться как Бог; в своих доличностных и сверхличностных "
     "проявлениях оно не вполне доступно человеческому пониманию. Божество характеризуется "
     "единством – актуальным или потенциальным – на всех сверхматериальных уровнях реальности; "
     "создания лучше всего постигают это объединяющее качество как божественность.",
     "synonyms + restructuring + active voice"),
    ("0:2.1", "paraphrase", False,
     "Эволюционирующие смертные существа неодолимо стремятся символически выразить свои "
     "конечные представления о Боге. Нравственное чувство долга и духовный идеализм человека "
     "составляют уровень ценностей – эмпирическую реальность, – которую трудно выразить "
     "символами.",
     "synonyms + nominalization changes"),
    ("0:5.8", "paraphrase", False,
     "2. Разум. Механизм мышления, восприятия и чувствования человеческого организма; "
     "совокупный сознательный и бессознательный опыт; интеллект, связанный с эмоциональной "
     "жизнью, который поднимается к уровню духа через поклонение и мудрость.",
     "sentence merging + reordering"),
    ("0:1.1", "paraphrase", False,
     "Вселенная вселенных демонстрирует феномены деятельности божества на разных уровнях "
     "космических реальностей, смыслов разума и духовных ценностей, но все эти служения – "
     "личностные или иные – согласованы божественным образом.",
     "passive/active, synonyms"),
    # ---- unchanged official translation: must NOT be flagged ---------------------------
    ("0:4.2", "unchanged", False, None, "official RU"),
    ("0:6.7", "unchanged", False, None, "official RU"),
    ("0:12.5", "unchanged", False, None, "official RU"),
    ("0:8.1", "unchanged", False, None, "official RU"),
    ("0:11.3", "unchanged", False, None, "official RU"),
    ("0:3.14", "unchanged", False, None, "official RU"),
]


def build():
    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {r["id"]: r for r in data}
    cases = []
    for n, (pid, ctype, expect, ru_text, note) in enumerate(CASES, start=1):
        rec = by_id[pid]
        cases.append({
            "id": f"{pid}::{ctype}::{n}",
            "para_id": pid,
            "issue_type": ctype,
            "expect_flagged": expect,
            "en": rec["en"],
            "ru_original": rec["ru"],
            "ru_test": ru_text if ru_text is not None else rec["ru"],
            "note": note,
        })
        assert expect is False or cases[-1]["ru_test"] != rec["ru"], f"{pid}: corruption equals original"
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(cases)} cases to {OUT_JSON}")


if __name__ == "__main__":
    build()
