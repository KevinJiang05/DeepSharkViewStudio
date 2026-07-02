"""Pair role helpers for front-priority seam candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PairRole:
    pair_id: str
    cameras: tuple[str, str]
    main_camera: str
    side_camera: str
    side_position: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_front_priority_pair_role(pair_id: str, cameras: list[str]) -> PairRole | None:
    if len(cameras) != 2 or "front" not in cameras:
        return None
    if cameras == ["front_left", "front"]:
        return PairRole(
            pair_id=pair_id,
            cameras=("front_left", "front"),
            main_camera="front",
            side_camera="front_left",
            side_position="left",
        )
    if cameras == ["front", "front_right"]:
        return PairRole(
            pair_id=pair_id,
            cameras=("front", "front_right"),
            main_camera="front",
            side_camera="front_right",
            side_position="right",
        )
    return None
