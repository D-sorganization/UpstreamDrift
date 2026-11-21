"""Cut MuJoCo skins into one piece per segment, for per-segment colour (#15150).

A MuJoCo skin takes one colour from its material, so a body drawn from three
skins (skin, shirt, trousers) cannot show one colour per segment. This module
cuts every ``<skin>`` of an MJCF tree into one skin per bone body. Each face
goes to the body that holds most of its three vertices' weight, and each piece
keeps every bone and weight of its vertices, so the drawn surface is unchanged.
Each piece gets its own untextured copy of the skin's material, which a
renderer can recolour. With ``border_method`` ``neighbour_majority`` the cut
follows the mesh faces and its border is straightened by neighbour-majority
rounds. With ``iso_line`` the faces between two segments are split along the
line where the two segments' weights are equal, so the border is a continuous
line through the faces. A vertex put on an edge takes the edge's position and
texture coordinate there and gets bones of its own, bound so that it stays on
the straight edge between the two skinned ends at every pose (blending the two
ends' weights instead would bow it off the edge). Settings are read from
``assets/body_skin_cc0/piece_settings.json``.
Appearance only; opt-in.

Limits: vertex normals at a cut edge come from one piece alone, so smooth
lighting can show the cut, and a skin with ``inflate`` (which moves vertices
along those normals) is not drawn exactly as before along it.
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.body_skin_fit import ASSET_DIR

__all__ = [
    "PieceSettings",
    "face_segments",
    "load_piece_settings",
    "split_mjcf_skins",
    "split_skins",
    "straighten_borders",
]

Ints = NDArray[np.int64]
Floats = NDArray[np.float64]
_EDGES = ((0, 1), (1, 2), (2, 0))  # the vertex slots of a triangle's three edges
_CORNERS = len(_EDGES)
_OWN = ("name", "material", "vertex", "face", "texcoord")  # rewritten per piece
_ISO, _MAJORITY = "iso_line", "neighbour_majority"


@dataclass(frozen=True)
class PieceSettings:
    """Settings of the cut (``piece_settings.json``; see its ``sources``)."""

    border_method: str
    border_rounds: int
    border_votes: int
    cut_smooth_rounds: int
    cut_snap_fraction: float
    weld_decimals: int
    number_digits: int
    dropped_material_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        """``TypeError`` for a non-integer, ``ValueError`` for one out of range."""
        whole = ("border_rounds", "border_votes", "weld_decimals", "number_digits")
        for name in (*whole, "cut_smooth_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.border_rounds < 0 or self.weld_decimals < 0 or self.number_digits < 1:
            raise ValueError("rounds and decimals must be >= 0, digits >= 1")
        if self.border_method not in (_ISO, _MAJORITY):
            raise ValueError(f"border_method must be {_ISO!r} or {_MAJORITY!r}")
        if self.cut_smooth_rounds < 0 or not 0.0 < self.cut_snap_fraction < 0.5:
            raise ValueError("smooth rounds must be >= 0, snap fraction in (0, 0.5)")
        if not _CORNERS / 2 < self.border_votes <= _CORNERS:
            raise ValueError("border_votes must be a majority of a face's three edges")


def load_piece_settings(path: Path | None = None) -> PieceSettings:
    """Read ``piece_settings.json``; raises ``ValueError`` for invalid values."""
    raw = json.loads((path or ASSET_DIR / "piece_settings.json").read_text())
    return PieceSettings(
        border_method=raw["border_method"],
        border_rounds=raw["border_rounds"],
        border_votes=raw["border_votes"],
        cut_smooth_rounds=raw["cut_smooth_rounds"],
        cut_snap_fraction=raw["cut_snap_fraction"],
        weld_decimals=raw["weld_decimals"],
        number_digits=raw["number_digits"],
        dropped_material_keys=tuple(raw["dropped_material_keys"]),
    )


def face_segments(faces: Ints, weights: Floats) -> Ints:
    """Per face, the column of ``weights`` holding most of its vertices' weight.

    Preconditions: ``faces`` is ``(F, 3)`` vertex indices into the ``(N, B)``
    ``weights`` (``ValueError`` otherwise). Ties go to the lower column.
    """
    faces, weights = np.asarray(faces, np.int64), np.asarray(weights, float)
    if faces.ndim != 2 or faces.shape[1] != _CORNERS or weights.ndim != 2:
        raise ValueError("faces must be (F, 3) and weights (N, B)")
    if faces.size and not 0 <= faces.min() <= faces.max() < len(weights):
        raise ValueError("a face names a vertex the weights do not have")
    labels: Ints = weights[faces].sum(axis=1).argmax(axis=1).astype(np.int64)
    return labels


def _edge_neighbours(vertices: Floats, faces: Ints, decimals: int) -> Ints:
    """Per face, the up to three faces across its edges (``-1`` where none).

    Vertices doubled at one place (a texture seam) are joined first, so a seam
    is no border. An edge shared by more than two faces joins none of them.
    """
    _, welded = np.unique(np.round(vertices, decimals), axis=0, return_inverse=True)
    corners = welded.ravel()[faces]
    edges = np.sort(corners[:, _EDGES], axis=2).reshape(-1, 2)
    owner = np.repeat(np.arange(len(faces)), _CORNERS)
    _, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    pairs = owner[order][counts[inverse[order]] == 2].reshape(-1, 2)
    both = np.concatenate([pairs, pairs[:, ::-1]])
    both = both[np.argsort(both[:, 0], kind="stable")]
    slot = np.arange(len(both)) - np.searchsorted(both[:, 0], both[:, 0])
    neighbours = np.full((len(faces), _CORNERS), -1, np.int64)
    neighbours[both[:, 0], slot] = both[:, 1]
    return neighbours


def straighten_borders(
    vertices: Floats, faces: Ints, labels: Ints, settings: PieceSettings
) -> Ints:
    """``labels`` with each face given the label most of its neighbours share.

    One round gives a face the label that ``border_votes`` of its edge
    neighbours share when it is not its own; rounds stop at ``border_rounds``
    or when nothing changes. Preconditions: ``vertices`` is ``(N, 3)``,
    ``faces`` ``(F, 3)`` and ``labels`` ``(F,)`` (``ValueError`` otherwise).
    Postcondition: a new array; no label appears that was not in ``labels``.
    """
    vertices, faces = np.asarray(vertices, float), np.asarray(faces, np.int64)
    labels = np.asarray(labels, np.int64).copy()
    if vertices.ndim != 2 or faces.ndim != 2 or labels.shape != (len(faces),):
        raise ValueError("vertices must be (N, 3), faces (F, 3) and labels (F,)")
    if not settings.border_rounds or not len(faces):
        return labels
    neighbours = _edge_neighbours(vertices, faces, settings.weld_decimals)
    for _ in range(settings.border_rounds):
        theirs = np.where(neighbours >= 0, labels[neighbours], -1)
        new = labels.copy()
        for column in range(_CORNERS):
            label = theirs[:, column]
            votes = (theirs == label[:, None]).sum(axis=1)
            win = (label >= 0) & (label != labels) & (votes >= settings.border_votes)
            new[win] = label[win]
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


@dataclass(frozen=True)
class _Skin:
    """One parsed ``<skin>``: its arrays and its bones' (element, ids, weights)."""

    node: ET.Element
    vertices: Floats
    faces: Ints
    texcoord: Floats | None
    bones: tuple[tuple[ET.Element, Ints, Floats], ...]


def _numbers(node: ET.Element, key: str, width: int, kind: Any = float) -> Any:
    text = node.get(key)
    if text is None:
        raise ValueError(f"skin {node.get('name')!r} lacks {key!r}")
    return np.array(text.split(), kind).reshape(-1, width)


def _read_skin(node: ET.Element) -> _Skin:
    vertices = _numbers(node, "vertex", _CORNERS)
    faces = _numbers(node, "face", _CORNERS, np.int64)
    texcoord = None if node.get("texcoord") is None else _numbers(node, "texcoord", 2)
    bones = []
    for bone in node.findall("bone"):
        ids = _numbers(bone, "vertid", 1, np.int64).ravel()
        share = _numbers(bone, "vertweight", 1).ravel()
        if len(ids) != len(share) or (ids.size and ids.max() >= len(vertices)):
            raise ValueError(f"skin {node.get('name')!r} has a bone off its vertices")
        bones.append((bone, ids, share))
    if not bones or not node.get("name"):
        raise ValueError("a skin to cut needs a name and at least one bone")
    return _Skin(node, vertices, faces, texcoord, tuple(bones))


def _text(values: Any, digits: int) -> str:
    return " ".join(format(float(v), f".{digits}g") for v in np.ravel(values))


def _smoothed(skin: _Skin, weights: Floats, settings: PieceSettings) -> Floats:
    """``weights`` averaged with each vertex's edge neighbours, once per round.

    Only the place of the cut is read from the result. Vertices doubled along a
    seam are joined first, so both copies get one value.
    """
    key = np.round(skin.vertices, settings.weld_decimals)
    _, welded = np.unique(key, axis=0, return_inverse=True)
    welded = welded.ravel()
    ends = np.sort(welded[skin.faces][:, _EDGES], axis=2).reshape(-1, 2)
    ends = np.unique(ends, axis=0)
    ends = np.concatenate([ends, ends[:, ::-1]])
    field = np.zeros((int(welded.max()) + 1, weights.shape[1]))
    np.add.at(field, welded, weights)
    field /= np.bincount(welded)[:, None]
    near = np.bincount(ends[:, 0], minlength=len(field))[:, None] + 1
    for _ in range(settings.cut_smooth_rounds):
        total = field.copy()
        np.add.at(total, ends[:, 0], field[ends[:, 1]])
        field = total / near
    return np.asarray(field[welded], float)


def _crossings(
    faces: Ints, field: Floats, snap: float
) -> tuple[Ints, Ints, Ints, Floats]:
    """Where the border crosses each face edge.

    Returns each vertex's segment (its largest ``field`` column), per face edge
    the vertex the border crosses it at (``-1`` where both ends share a
    segment), and the new vertices as ``(M, 3)`` old vertices and their shares.
    A crossing within ``snap`` of an end of its edge is that end, so no sliver
    is cut. An edge is read from its lower segment to its higher one, so both
    copies of an edge doubled along a seam cross at one place.
    """
    labels = field.argmax(axis=1)
    ends = faces[:, _EDGES]
    turned = labels[ends[..., 0]] > labels[ends[..., 1]]
    ends = np.where(turned[..., None], ends[..., ::-1], ends).reshape(-1, 2)
    pairs, edge = np.unique(ends, axis=0, return_inverse=True)
    first, second = pairs[:, 0], pairs[:, 1]
    low, high = labels[first], labels[second]
    lead = field[first, low] - field[first, high]
    trail = field[second, high] - field[second, low]
    span = lead + trail
    along = np.where(span > 0, lead / np.where(span > 0, span, 1.0), 0.5)
    at = np.full(len(pairs), -1, np.int64)
    crossed = low != high
    at[crossed & (along < snap)] = first[crossed & (along < snap)]
    at[crossed & (along > 1 - snap)] = second[crossed & (along > 1 - snap)]
    fresh = crossed & (at < 0)
    at[fresh] = len(field) + np.arange(int(fresh.sum()))
    corners = np.column_stack([first, second, second])[fresh]
    share = np.column_stack([1 - along, along, np.zeros(len(along))])[fresh]
    return labels, at[edge.reshape(-1, _CORNERS)], corners, share


def _least_area(vertices: Floats, faces: Ints) -> Floats:
    """Per row of ``(K, T, 3)`` triangles, the area of its smallest one."""
    a, b, c = (vertices[faces[..., k]] for k in range(_CORNERS))
    areas: Floats = np.linalg.norm(np.cross(b - a, c - a), axis=-1).min(axis=-1)
    return areas


def _cut_faces(
    vertices: Floats, faces: Ints, labels: Ints, cross: Ints
) -> tuple[Ints, Ints, Ints]:
    """Faces split along their crossings: new faces, their segments, centres.

    A face in one segment stays. A face in two gives its lone corner a triangle
    and the other two a quadrilateral, halved along the diagonal that leaves the
    larger smallest triangle. A face in three is fanned about a new centre
    vertex (index ``len(vertices)`` onward; returned as the faces they centre).
    Triangles that a snapped crossing closed to a line are left out.
    """
    mine = labels[faces]
    same = mine == np.roll(mine, -1, axis=1)  # corner k and the next share one
    lone = np.where(same[:, 1], 0, np.where(same[:, 2], 1, 2))
    turn = (lone[:, None] + np.arange(_CORNERS)) % _CORNERS
    rows = np.arange(len(faces))[:, None]
    v, c, part = faces[rows, turn], cross[rows, turn], mine[rows, turn]
    kinds = _CORNERS - same.sum(axis=1)  # 0: one segment, 2: two, 3: three
    two, three = kinds == 2, np.flatnonzero(kinds == _CORNERS)
    v0, v1, v2, p, q = (x[two] for x in (v[:, 0], v[:, 1], v[:, 2], c[:, 0], c[:, 2]))
    wide = np.array([[p, v1, q], [v1, v2, q]]).transpose(2, 0, 1)
    tall = np.array([[p, v1, v2], [p, v2, q]]).transpose(2, 0, 1)
    pick = _least_area(vertices, wide) >= _least_area(vertices, tall)
    halves = np.where(pick[:, None, None], wide, tall)
    centre = len(vertices) + np.arange(len(three))
    fan, fan_part = [], []
    for k in range(_CORNERS):
        corner, after, before = v[three, k], c[three, k], c[three, k - 1]
        fan += [np.stack([corner, after, centre], 1)]
        fan += [np.stack([corner, centre, before], 1)]
        fan_part += [part[three, k]] * 2
    cut = [faces[kinds == 0], np.stack([v0, p, q], 1), halves[:, 0], halves[:, 1]]
    segments = [mine[kinds == 0, 0], part[two, 0], part[two, 1], part[two, 1]]
    out = np.concatenate(cut + fan)
    out_part = np.concatenate(segments + fan_part)
    real = (out != np.roll(out, -1, axis=1)).all(axis=1)
    return out[real], out_part[real], faces[three]


def _iso_cut(
    skin: _Skin, weights: Floats, settings: PieceSettings
) -> tuple[_Skin, Ints]:
    """``skin`` with its faces split where two segments' weights are equal.

    Returns the refined skin and each of its faces' segment. Postconditions:
    the old vertices and bones keep their indices; every new vertex is a fixed
    weighted mean of the corners of one old face, in texture coordinate and, by
    one bone of its own per bone of those corners, in position at every pose;
    so the new faces cover the old ones exactly, posed or not.
    """
    field = _smoothed(skin, weights, settings)
    labels, cross, corners, share = _crossings(
        skin.faces, field, settings.cut_snap_fraction
    )

    def blend(values: Floats) -> Floats:
        mix = share.reshape(share.shape + (1,) * (values.ndim - 1))
        return np.concatenate([values, (values[corners] * mix).sum(axis=1)])

    count = len(skin.vertices)
    faces, parts, centred = _cut_faces(blend(skin.vertices), skin.faces, labels, cross)
    corners = np.concatenate([corners, centred])
    share = np.concatenate([share, np.full(centred.shape, 1 / _CORNERS)])
    rest = blend(skin.vertices)
    bones = list(skin.bones)
    for bone, ids, held in skin.bones:
        dense = np.zeros(count)
        np.add.at(dense, ids, held)
        pull = dense[corners] * share
        weight = pull.sum(axis=1)
        bind = _numbers(bone, "bindpos", _CORNERS).ravel()
        for new in np.flatnonzero(weight):
            rides = pull[new] @ skin.vertices[corners[new]] / weight[new]
            at = _text(bind + rest[count + new] - rides, settings.number_digits)
            own = ET.Element("bone", {**dict(bone.items()), "bindpos": at})
            bones.append((own, np.array([count + new]), weight[new : new + 1]))
    texcoord = None if skin.texcoord is None else blend(skin.texcoord)
    return _Skin(skin.node, rest, faces, texcoord, tuple(bones)), parts


def _piece(skin: _Skin, faces: Ints, name: str, material: str, digits: int) -> Any:
    """The ``<skin>`` of ``faces``, with every bone share of its vertices."""
    keep = np.unique(faces)
    remap = np.full(len(skin.vertices), -1, np.int64)
    remap[keep] = np.arange(len(keep))
    attributes = {k: v for k, v in skin.node.items() if k not in _OWN}
    piece = ET.Element("skin", {**attributes, "name": name, "material": material})
    piece.set("vertex", _text(skin.vertices[keep], digits))
    piece.set("face", " ".join(str(int(i)) for i in remap[faces].ravel()))
    if skin.texcoord is not None:
        piece.set("texcoord", _text(skin.texcoord[keep], digits))
    for bone, ids, share in skin.bones:
        held = remap[ids] >= 0
        if held.any():
            kept = {k: v for k, v in bone.items() if not k.startswith("vert")}
            node = ET.SubElement(piece, "bone", kept)
            node.set("vertid", " ".join(str(int(i)) for i in remap[ids[held]]))
            node.set("vertweight", _text(share[held], digits))
    return piece


def _piece_material(
    root: ET.Element, skin: ET.Element, index: int, dropped: tuple[str, ...]
) -> str:
    """Add an untextured copy of ``skin``'s material to its asset; its name."""
    wanted = skin.get("material")
    for asset in root.iter("asset"):
        for material in asset.findall("material"):
            if material.get("name") == wanted:
                own = copy.deepcopy(material)
                own.set("name", f"{wanted}__seg__{skin.get('name')}__{index}")
                for key in dropped:
                    own.attrib.pop(key, None)
                asset.append(own)
                return str(own.get("name"))
    raise ValueError(f"skin {skin.get('name')!r} has no material to recolour")


def split_skins(
    root: ET.Element, settings: PieceSettings | None = None
) -> dict[str, str]:
    """Replace every ``<skin>`` under ``root`` by one skin per bone body.

    Preconditions: each skin has a name, a material defined in an ``<asset>``,
    ``vertex``, ``face`` and bones whose ``vertid`` are its vertices
    (``ValueError`` otherwise). Postcondition: returns piece skin name -> body,
    in document order; every face is in exactly one piece, each piece
    ``<skin>__seg<n>`` stands where its skin stood and wears its own material
    ``<material>__seg__<skin>__<n>``; nothing else in ``root`` changes.
    """
    settings = settings or load_piece_settings()
    table: dict[str, str] = {}
    homes = [(p, s) for p in root.iter() for s in p.findall("skin")]
    for parent, node in homes:
        skin = _read_skin(node)
        bodies = sorted({str(bone.get("body")) for bone, _, _ in skin.bones})
        weights = np.zeros((len(skin.vertices), len(bodies)))
        for bone, ids, share in skin.bones:
            np.add.at(weights[:, bodies.index(str(bone.get("body")))], ids, share)
        if settings.border_method == _ISO:
            skin, labels = _iso_cut(skin, weights, settings)
        else:
            labels = face_segments(skin.faces, weights)
            labels = straighten_borders(skin.vertices, skin.faces, labels, settings)
        at = list(parent).index(node)
        parent.remove(node)
        for index, body in enumerate(bodies):
            faces = skin.faces[labels == index]
            if not len(faces):
                continue
            material = _piece_material(
                root, node, index, settings.dropped_material_keys
            )
            name = f"{node.get('name')}__seg{index}"
            piece = _piece(skin, faces, name, material, settings.number_digits)
            parent.insert(at, piece)
            at += 1
            table[name] = body
    return table


def split_mjcf_skins(
    xml: str, settings: PieceSettings | None = None
) -> tuple[str, dict[str, str]]:
    """:func:`split_skins` of an MJCF string: the new MJCF and the piece table."""
    from defusedxml import ElementTree as SafeET

    root = SafeET.fromstring(xml)
    table = split_skins(root, settings)
    return ET.tostring(root, encoding="unicode"), table
