# Бенчмарк анализа перевода

Датасет: `bench/translation/dataset.json` — 17 реальных абзацев **Paper 0 (Предисловие)**: 7 намеренных искажений (пропуск, добавление, искажение смысла, неверный термин, инверсия отрицания, неверное число, частичный пропуск), 4 допустимых перефразирования (синонимы, перестановка, залог, слияние предложений) и 6 официальных переводов без изменений. Все варианты написаны вручную (`build_dataset.py`), чтобы каждое искажение было реальным, а каждое перефразирование — смысл-сохраняющим.

«Отмечено» = после всего конвейера остался хотя бы один **critical/major**-пункт (minor учитывается отдельно как `any`). Это тот же код, что и в рабочем анализе (`app.llm.translation.review_batch`).

## Модель `qwen3.6-35b-a3b-mtp@q5_k_m`, рассуждение: off

| Конфигурация | Precision | Recall | F1 | TP/FP/TN/FN | с/абзац | вызовов | ток. (prompt→compl.) |
|---|---|---|---|---|---|---|---|
| v1: исходный промпт, без проверки | **1.00** | **0.43** | 0.60 | 3/0/10/4 | 14.2 | 1.0 | 9550→4016 |
| v2: промпт с доказательствами + проверка цитат | **1.00** | **0.86** | 0.92 | 6/0/10/1 | 6.0 | 1.0 | 23144→959 |
| v2v: v2 + верификационный проход | **1.00** | **0.86** | 0.92 | 6/0/10/1 | 8.4 | 1.41 | 25646→1301 |

### v1: исходный промпт, без проверки

| Тип | Кейсов | Отмечено (critical/major) | Любой пункт (в т.ч. minor) |
|---|---|---|---|
| omission | 1 | 0 | 0 |
| addition | 1 | 0 | 0 |
| distortion | 1 | 0 | 0 |
| wrong_term | 1 | 1 | 1 |
| negation_flip | 1 | 1 | 1 |
| wrong_number | 1 | 1 | 1 |
| partial_omission | 1 | 0 | 1 |
| paraphrase | 4 | 0 | 1 |
| unchanged | 6 | 0 | 5 |

<details><summary>Построчно</summary>

| Кейс | Ожидалось | Отмечено | Пункты | с | вызовов |
|---|---|---|---|---|---|
| 0:1.1::omission::1 | True | False | — | 10.4 | 1 |
| 0:1.17::addition::2 | True | False | — | 9.7 | 1 |
| 0:3.9::distortion::3 | True | False | — | 9.8 | 1 |
| 0:1.14::wrong_term::4 | True | True | critical: The term 'Paradise Trinity' is translated as 'Земная Троица' (Earthly ; major: 'Experiential' is translated as 'эмпирическим'. In the context of Uran | 17.8 | 1 |
| 0:6.6::negation_flip::5 | True | True | critical: The translation reverses the logical relationship between the energies | 10.5 | 1 |
| 0:3.14::wrong_number::6 | True | True | critical: Incorrect translation of 'Seven Master Spirits' as 'Двенадцати Главных | 11.9 | 1 |
| 0:2.2::partial_omission::7 | True | False | minor: Omission of 'one and only' modifier for First Cause; minor: Slight awkwardness in 'относительным выражением божественности' | 15.7 | 1 |
| 0:1.2::paraphrase::8 | False | False | — | 14.7 | 1 |
| 0:2.1::paraphrase::9 | False | False | — | 10.3 | 1 |
| 0:5.8::paraphrase::10 | False | False | minor: The phrase 'reaching upward through worship and wisdom to the spirit l | 16.5 | 1 |
| 0:1.1::paraphrase::11 | False | False | — | 15.0 | 1 |
| 0:4.2::unchanged::12 | False | False | minor: The phrase 'nonpersonalizable values' is translated as 'неперсонализир | 11.9 | 1 |
| 0:6.7::unchanged::13 | False | False | minor: Translation of 'Universe power' as 'Энергия вселенной'; minor: Translation of 'directly responsive' as 'непосредственно подчиняются' | 17.2 | 1 |
| 0:12.5::unchanged::14 | False | False | — | 14.7 | 1 |
| 0:8.1::unchanged::15 | False | False | minor: Translation of 'atone for' as 'возместить'; minor: Translation of 'creature' as 'создание'; minor: Translation of 'sevenfold approach' as 'семичастное приближение' | 22.9 | 1 |
| 0:11.3::unchanged::16 | False | False | minor: The term 'supposedly' is translated as 'предположительно', which impli | 17.3 | 1 |
| 0:3.14::unchanged::17 | False | False | minor: The term 'factualizing' is translated as 'фактуализирующийся', which i | 14.5 | 1 |

</details>

### v2: промпт с доказательствами + проверка цитат

| Тип | Кейсов | Отмечено (critical/major) | Любой пункт (в т.ч. minor) |
|---|---|---|---|
| omission | 1 | 1 | 1 |
| addition | 1 | 1 | 1 |
| distortion | 1 | 1 | 1 |
| wrong_term | 1 | 1 | 1 |
| negation_flip | 1 | 1 | 1 |
| wrong_number | 1 | 1 | 1 |
| partial_omission | 1 | 0 | 0 |
| paraphrase | 4 | 0 | 0 |
| unchanged | 6 | 0 | 0 |

<details><summary>Построчно</summary>

| Кейс | Ожидалось | Отмечено | Пункты | с | вызовов |
|---|---|---|---|---|---|
| 0:1.1::omission::1 | True | True | major: The entire second clause about divine coordination is missing from the | 11.6 | 1 |
| 0:1.17::addition::2 | True | True | major: RU adds a sentence claiming these qualities are accessible only to hig | 7.6 | 1 |
| 0:3.9::distortion::3 | True | True | critical: The original states limitation is by 'volition', but the translation i | 9.1 | 1 |
| 0:1.14::wrong_term::4 | True | True | critical: 'Experiential' is translated as 'empirical', which changes the meaning; critical: 'Paradise' is translated as 'Earthly' (Земной), which contradicts the  | 15.0 | 1 |
| 0:6.6::negation_flip::5 | True | True | critical: Negation and responsiveness are swapped: EN says responsive to Paradis | 10.6 | 1 |
| 0:3.14::wrong_number::6 | True | True | major: The number 'Seven' is incorrectly translated as 'Twelve'. | 7.7 | 1 |
| 0:2.2::partial_omission::7 | True | False | — | 3.7 | 1 |
| 0:1.2::paraphrase::8 | False | False | — | 3.8 | 1 |
| 0:2.1::paraphrase::9 | False | False | — | 3.7 | 1 |
| 0:5.8::paraphrase::10 | False | False | — | 3.7 | 1 |
| 0:1.1::paraphrase::11 | False | False | — | 3.7 | 1 |
| 0:4.2::unchanged::12 | False | False | — | 3.5 | 1 |
| 0:6.7::unchanged::13 | False | False | — | 3.6 | 1 |
| 0:12.5::unchanged::14 | False | False | — | 3.6 | 1 |
| 0:8.1::unchanged::15 | False | False | — | 3.5 | 1 |
| 0:11.3::unchanged::16 | False | False | — | 3.6 | 1 |
| 0:3.14::unchanged::17 | False | False | — | 3.6 | 1 |

</details>

### v2v: v2 + верификационный проход

| Тип | Кейсов | Отмечено (critical/major) | Любой пункт (в т.ч. minor) |
|---|---|---|---|
| omission | 1 | 1 | 1 |
| addition | 1 | 1 | 1 |
| distortion | 1 | 1 | 1 |
| wrong_term | 1 | 1 | 1 |
| negation_flip | 1 | 1 | 1 |
| wrong_number | 1 | 1 | 1 |
| partial_omission | 1 | 0 | 0 |
| paraphrase | 4 | 0 | 0 |
| unchanged | 6 | 0 | 0 |

<details><summary>Построчно</summary>

| Кейс | Ожидалось | Отмечено | Пункты | с | вызовов |
|---|---|---|---|---|---|
| 0:1.1::omission::1 | True | True | major: The entire second clause about divine coordination and personal/impers | 15.2 | 2 |
| 0:1.17::addition::2 | True | True | major: RU adds a sentence about accessibility to higher races, absent from EN | 12.3 | 2 |
| 0:3.9::distortion::3 | True | True | critical: RU replaces 'volition' with 'laws of matter', completely changing the  | 15.2 | 2 |
| 0:1.14::wrong_term::4 | True | True | critical: 'Paradise' translated as 'Земной' (Earthly) instead of 'Райской', reve | 28.3 | 3 |
| 0:6.6::negation_flip::5 | True | True | critical: Negations and responses are swapped: EN says responsive to Paradise/un | 16.9 | 2 |
| 0:3.14::wrong_number::6 | True | True | major: The number 'Seven' is incorrectly translated as 'Twelve'. | 13.8 | 2 |
| 0:2.2::partial_omission::7 | True | False | — | 4.1 | 1 |
| 0:1.2::paraphrase::8 | False | False | — | 3.8 | 1 |
| 0:2.1::paraphrase::9 | False | False | — | 3.7 | 1 |
| 0:5.8::paraphrase::10 | False | False | — | 3.6 | 1 |
| 0:1.1::paraphrase::11 | False | False | — | 3.6 | 1 |
| 0:4.2::unchanged::12 | False | False | — | 3.5 | 1 |
| 0:6.7::unchanged::13 | False | False | — | 3.6 | 1 |
| 0:12.5::unchanged::14 | False | False | — | 3.6 | 1 |
| 0:8.1::unchanged::15 | False | False | — | 3.6 | 1 |
| 0:11.3::unchanged::16 | False | False | — | 3.6 | 1 |
| 0:3.14::unchanged::17 | False | False | — | 3.7 | 1 |

</details>

## История и выводы

- **Первый прогон** (gemma-4-26b-a4b-it-qat, рассуждение выкл., старый промпт v1, предыдущий датасет из случайных абзацев): Precision 0.45, Recall 1.00 (TP=5, FP=6): модель отмечала практически всё, включая перефразирования и неизменённые пары (но и «recall 1.00» был завышен: кейс wrong_term в том датасете совпадал с оригиналом). С включённым рассуждением (по умолчанию модели) при max_tokens=1500 всё бюджет уходил в reasoning_content (1497 ток.), JSON-ответа не оставалось: 0/5 обнаружений, ~58 с на пару вместо ~8 с.
- Поэтому по умолчанию рассуждение для анализа перевода выключено (`reasoning_effort=none`), а точность повышается промптом v2 (доказательства-цитаты, список «не ошибок», калибровочные примеры, «[] — нормальный ответ»), проверкой цитат в коде и верификационным проходом.
