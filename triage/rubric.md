# Triage severity rubric (Step 1.1)

Code: `triage/rubric.py` (`rubric_score(features) -> (severity, points, reasons)`).
Status: **Approved by user (Step 1.1).**

## How it works
1. Start with the issue type's base points.
2. Add modifier points (each one becomes a reason in the explanation).
3. Map total points to a `Severity` level. The SLA table (Step 0.4) then turns severity into a
   deadline per department, so there is no separate "urgency" output.
4. Safety floor: if an accident is reported, severity is at least **high**.

### Base points
| Issue type     | Points | Why |
|----------------|-------:|-----|
| pothole        | 2 | direct injury risk to two-wheelers |
| water_leak     | 2 | wastes supply; undermines roads |
| drain_blockage | 2 | monsoon flooding, mosquito breeding |
| streetlight    | 1 | safety at night, not immediate |
| garbage        | 1 | nuisance, health over days |
| damaged_sign   | 1 | low unless it causes accidents |
| other          | 1 | unknown; let modifiers decide |

### Modifiers
| Condition                                  | Points |
|--------------------------------------------|-------:|
| road_type == main                          | +1 |
| school within 200 m                        | +1 |
| hospital within 300 m                      | +1 |
| 3-9 other reports merged (duplicate_count) | +1 |
| 10+ other reports merged                   | +2 |
| text mentions children                     | +1 |
| text mentions accident                     | +2 |
| text mentions flooding                     | +2 |

### Points -> severity
| Points | Severity |
|-------:|----------|
| 0-2 | low |
| 3-4 | medium |
| 5-6 | high |
| 7+  | critical |

## Worked examples (tested in `triage/tests/test_rubric.py`)
| # | Case | Points | Severity | Explanation |
|---|------|-------:|----------|-------------|
| 1 | Pothole on a main road in Karol Bagh, 150 m from a school, complaint mentions children | 2+1+1+1 = 5 | high | A pothole on a main road next to a school where children are mentioned is a likely injury. |
| 2 | Streetlight out on a residential road in Patel Nagar, nothing else | 1 | low | A single streetlight with no nearby risk can wait for routine repair. |
| 3 | Garbage overflow 250 m from a Daryaganj hospital, 4 people reported it | 1+1+1 = 3 | medium | Garbage near a hospital that many people have reported is a health concern within a day or two. |
| 4 | Blocked drain on a main road near Minto Bridge, flooding, 12 reports | 2+1+2+2 = 7 | critical | A flooded main road reported by many people blocks traffic and endangers people now. |
| 5 | Damaged sign on a main road, complaint mentions an accident | 1+1+2 = 4 -> floor | high | Points say medium, but a reported accident is never below high. |
| 6 | Water leak in a lane in Chandni Chowk, 120 m from a school | 2+1 = 3 | medium | A leak next to a school needs fixing soon but is not dangerous yet. |

## Effect on the corpus (2400 non-spam incidents, Step 0.5)
| Severity | Count | Share |
|----------|------:|------:|
| low      | 1361 | 56.7% |
| medium   |  730 | 30.4% |
| high     |  297 | 12.4% |
| critical |   12 |  0.5% |

Critical is rare by design (it needs several risks at once). Step 1.2 should oversample
high-risk cases so the model sees enough critical examples.
