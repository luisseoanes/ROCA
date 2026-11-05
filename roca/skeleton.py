"""Definición de puntos clave (keypoints) de caballo y jinete.

El esqueleto del caballo es un subconjunto compatible con los marcadores
anatómicos habituales (y mapeable a las articulaciones de hSMAL / PFERD).
El del jinete sigue un subconjunto de COCO-WholeBody / SMPL-X.
"""

HORSE_KP = [
    "nose", "eye", "poll", "throat", "withers", "chest", "mid_back", "croup",
    "tail_base", "tail_tip",
    "LF_shoulder", "LF_knee", "LF_fetlock", "LF_hoof",
    "RF_shoulder", "RF_knee", "RF_fetlock", "RF_hoof",
    "LH_hip", "LH_hock", "LH_fetlock", "LH_hoof",
    "RH_hip", "RH_hock", "RH_fetlock", "RH_hoof",
]

HORSE_EDGES = [
    ("nose", "eye"), ("eye", "poll"), ("poll", "throat"), ("throat", "nose"),
    ("poll", "withers"), ("throat", "chest"), ("withers", "chest"),
    ("withers", "mid_back"), ("mid_back", "croup"), ("croup", "tail_base"),
    ("tail_base", "tail_tip"),
    ("withers", "LF_shoulder"), ("LF_shoulder", "LF_knee"), ("LF_knee", "LF_fetlock"), ("LF_fetlock", "LF_hoof"),
    ("withers", "RF_shoulder"), ("RF_shoulder", "RF_knee"), ("RF_knee", "RF_fetlock"), ("RF_fetlock", "RF_hoof"),
    ("croup", "LH_hip"), ("LH_hip", "LH_hock"), ("LH_hock", "LH_fetlock"), ("LH_fetlock", "LH_hoof"),
    ("croup", "RH_hip"), ("RH_hip", "RH_hock"), ("RH_hock", "RH_fetlock"), ("RH_fetlock", "RH_hoof"),
    ("chest", "LF_shoulder"), ("chest", "RF_shoulder"), ("LH_hip", "RH_hip"),
]

RIDER_KP = [
    "head", "neck", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_hand", "R_hand", "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle",
]

RIDER_EDGES = [
    ("head", "neck"), ("neck", "L_shoulder"), ("neck", "R_shoulder"),
    ("L_shoulder", "L_elbow"), ("L_elbow", "L_hand"),
    ("R_shoulder", "R_elbow"), ("R_elbow", "R_hand"),
    ("L_shoulder", "L_hip"), ("R_shoulder", "R_hip"), ("L_hip", "R_hip"),
    ("L_hip", "L_knee"), ("L_knee", "L_ankle"), ("R_hip", "R_knee"), ("R_knee", "R_ankle"),
]

LEGS = ["LF", "RF", "LH", "RH"]
HOOVES = [f"{leg}_hoof" for leg in LEGS]


def hidx(name: str) -> int:
    return HORSE_KP.index(name)


def ridx(name: str) -> int:
    return RIDER_KP.index(name)


def edges_idx(names: list[str], edges: list[tuple[str, str]]) -> list[list[int]]:
    return [[names.index(a), names.index(b)] for a, b in edges]
