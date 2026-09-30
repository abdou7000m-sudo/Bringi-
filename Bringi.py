from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


try:
    from zone import bbox_in_polygon, draw_zone_polygon
except ImportError:

    def bbox_in_polygon(polygon, bbox, anchor="bottom") -> bool:
        pt = _anchor_xy(bbox, anchor)
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        return (
            cv2.pointPolygonTest(
                pts,
                (float(pt[0]), float(pt[1])),
                False,
            )
            >= 0
        )

    def draw_zone_polygon(
        frame,
        polygon,
        *,
        color,
        alpha=0.2,
        thickness=2,
    ):
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)

        frame = cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1.0 - alpha,
            0,
        )

        cv2.polylines(
            frame,
            [pts],
            True,
            color,
            thickness,
            cv2.LINE_AA,
        )

        return frame


try:
    from ultravis.trackers import Track
except ImportError:

    @dataclass
    class Track:
        track_id: int
        bbox: tuple[float, float, float, float]
        class_id: int = 0
        class_name: str = "person"
        confidence: float = 1.0
        is_confirmed: bool = True


logger = logging.getLogger(__name__)

DEFAULT_PERSON_CLASSES = ("person",)

LEVEL_NEAR = "near"
LEVEL_DANGER = "danger"
LEVEL_CLEAR = "clear"

NEAR_COLOR = (0, 180, 255)
DANGER_COLOR = (0, 0, 255)
CLEAR_COLOR = (40, 170, 70)

HUD_HEIGHT = 100
HUD_WIDTH = 720


@dataclass
class RiverbankEvent:
    track_id: int
    kind: str
    level: str
    frame_index: int
    class_id: int
    class_name: str = ""
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
    )


@dataclass
class RiverbankMonitor:

    near_polygon: list[tuple[int, int]]
    danger_polygon: list[tuple[int, int]]

    classes: tuple[str, ...] = DEFAULT_PERSON_CLASSES

    near_s: float = 3.0
    danger_s: float = 1.0

    max_speed_px_s: float = 40.0
    loiter_radius_px: float = 100.0

    grace_frames: int = 15
    anchor: str = "bottom"
    fps: float = 10.0

    _class_set: set[str] = field(
        init=False,
        repr=False,
    )

    # Zone entry time
    _near_since: dict[int, float] = field(
        default_factory=dict
    )

    _danger_since: dict[int, float] = field(
        default_factory=dict
    )

    # Loiter timers
    _loiter_near: dict[int, float] = field(
        default_factory=dict
    )

    _loiter_danger: dict[int, float] = field(
        default_factory=dict
    )

    # Local play positions
    _home_near: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

    _home_danger: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

    # Previous center
    _center: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

    # Alert states
    _alerted_near: set[int] = field(
        default_factory=set
    )

    _alerted_danger: set[int] = field(
        default_factory=set
    )

    # Missing-frame grace
    _grace_near: dict[int, int] = field(
        default_factory=dict
    )

    _grace_danger: dict[int, int] = field(
        default_factory=dict
    )

    # Last known track
    _last_track: dict[int, Track] = field(
        default_factory=dict
    )

    # UI state
    _remaining: dict[int, float] = field(
        default_factory=dict
    )

    _label: dict[int, str] = field(
        default_factory=dict
    )

    _prev_now: float | None = None

    _frame_index: int = 0

    def __post_init__(self) -> None:

        self.near_s = max(0.0, float(self.near_s))
        self.danger_s = max(0.0, float(self.danger_s))

        self.max_speed_px_s = max(
            0.0,
            float(self.max_speed_px_s),
        )



        self.loiter_radius_px = max(
            1.0,
            float(self.loiter_radius_px),
        )

        self.grace_frames = max(
            1,
            int(self.grace_frames),
        )

        self.fps = max(
            1.0,
            float(self.fps),
        )

        self._class_set = {
            str(name).strip().lower()
            for name in self.classes
            if str(name).strip()
        }

        if not self._class_set:
            self._class_set = set(
                DEFAULT_PERSON_CLASSES
            )

# part 2 screen 1     

    def update(
        self,
        tracks: list[Track],
        *,
        now: float | None = None,
        frame_index: int | None = None,
    ) -> list[RiverbankEvent]:

        if frame_index is None:
            frame_index = self._frame_index

        # Important:
        # Keep internal frame counter synchronized even when
        # frame_index is supplied externally.
        self._frame_index = int(frame_index) + 1

        if now is None:
            now = frame_index / self.fps

        now = float(now)

        if self._prev_now is None:
            dt = 1.0 / self.fps
        else:
            dt = now - self._prev_now

            # Protect against timestamps going backwards
            # or very large gaps.
            dt = max(
                1e-3,
                min(dt, 1.0),
            )

        self._prev_now = now

        present_near: dict[int, Track] = {}
        present_danger: dict[int, Track] = {}

        for track in tracks:

            if not track.is_confirmed:
                continue

            class_name = (
                track.class_name or ""
            ).strip().lower()

            if class_name not in self._class_set:
                continue

            in_danger = bbox_in_polygon(
                self.danger_polygon,
                track.bbox,
                self.anchor,
            )

            in_near = bbox_in_polygon(
                self.near_polygon,
                track.bbox,
                self.anchor,
            )

            # Ignore tracks outside both zones.
            if not in_danger and not in_near:
                continue

            track_id = track.track_id

            self._last_track[track_id] = track

            # Danger has priority.
            if in_danger:
                present_danger[track_id] = track

            elif in_near:
                present_near[track_id] = track

        events: list[RiverbankEvent] = []

        events.extend(
            self._tick_present(
                present_near,
                present_danger,
                now,
                dt,
                frame_index,
            )
        )

        events.extend(
            self._tick_missing_danger(
                present_danger,
                frame_index,
            )
        )

        events.extend(
            self._tick_missing_near(
                present_near,
                present_danger,
                frame_index,
            )
        )

        return events

    # part 4 screen 3

    def level_of(
        self,
        bbox: tuple[float, float, float, float],
    ) -> str | None:

        # Danger always wins if polygons overlap.
        if bbox_in_polygon(
            self.danger_polygon,
            bbox,
            self.anchor,
        ):
            return LEVEL_DANGER

        if bbox_in_polygon(
            self.near_polygon,
            bbox,
            self.anchor,
        ):
            return LEVEL_NEAR

        return None

    # part 5 screen 4
             
    def _speed_and_center(
        self,
        track_id: int,
        bbox: tuple[float, float, float, float],
        dt: float,
    ) -> tuple[float, tuple[float, float]]:

        center = _anchor_xy(
            bbox,
            "center",
        )

        previous = self._center.get(track_id)

        speed = 0.0

        if previous is not None:
            distance = math.hypot(
                center[0] - previous[0],
                center[1] - previous[1],
            )

            speed = distance / max(
                1e-3,
                dt,
            )

        self._center[track_id] = center

        return speed, center

    # part 6 screen 5

    def _accumulate_loiter(
        self,
        track_id: int,
        *,
        center: tuple[float, float],
        speed: float,
        dt: float,
        loiter: dict[int, float],
        home: dict[int, tuple[float, float]],
    ) -> tuple[float, bool]:

        # Fast movement = walking through.
        moving = (
            self.max_speed_px_s > 0
            and speed > self.max_speed_px_s
        )

        if moving:

            # IMPORTANT:
            # Reset the actual dwell timer.
            loiter[track_id] = 0.0

            # New play location will be established
            # after the person stops.
            home.pop(track_id, None)

            return 0.0, False

        spot = home.get(track_id)

        # First stationary frame.
        if spot is None:

            home[track_id] = center

            loiter[track_id] = dt

            return loiter[track_id], True

        distance = math.hypot(
            center[0] - spot[0],
            center[1] - spot[1],
        )

        # Person moved too far from the local play spot.
        if distance > self.loiter_radius_px:

            home[track_id] = center

            loiter[track_id] = dt

            return loiter[track_id], True

        # Still in the same play area.
        loiter[track_id] = (
            loiter.get(track_id, 0.0) + dt
        )

        return loiter[track_id], True

    # part 7 screen 6

    def _tick_present(
        self,
        present_near: dict[int, Track],
        present_danger: dict[int, Track],
        now: float,
        dt: float,
        frame_index: int,
    ) -> list[RiverbankEvent]:

        events: list[RiverbankEvent] = []

        active_ids = (
            set(present_near)
            | set(present_danger)
        )

        for track_id in active_ids:

            # Danger has priority.
            track = (
                present_danger.get(track_id)
                or present_near[track_id]
            )

            in_danger = track_id in present_danger
            in_near = track_id in present_near

            speed, center = self._speed_and_center(
                track_id,
                track.bbox,
                dt,
            )

    # part 8 screen 7

            if in_danger:

                self._grace_danger.pop(
                    track_id,
                    None,
                )

                if track_id not in self._danger_since:
                    self._danger_since[track_id] = now

            if in_near:

                self._grace_near.pop(
                    track_id,
                    None,
                )

                if track_id not in self._near_since:
                    self._near_since[track_id] = now

    # part 9 screen 8

            near_loiter = 0.0
            near_playing = False

            if in_near and not in_danger:

                near_loiter, near_playing = (
                    self._accumulate_loiter(
                        track_id,
                        center=center,
                        speed=speed,
                        dt=dt,
                        loiter=self._loiter_near,
                        home=self._home_near,
                    )
                )

            else:

                self._loiter_near.pop(
                    track_id,
                    None,
                )

                self._home_near.pop(
                    track_id,
                    None,
                )

    # part 10 screen 9

            danger_loiter = 0.0
            danger_playing = False

            if in_danger:

                danger_loiter, danger_playing = (
                    self._accumulate_loiter(
                        track_id,
                        center=center,
                        speed=speed,
                        dt=dt,
                        loiter=self._loiter_danger,
                        home=self._home_danger,
                    )
                )

            else:

                self._loiter_danger.pop(
                    track_id,
                    None,
                )

                self._home_danger.pop(
                    track_id,
                    None,
                )

    # part 11 screen 10            

            remain_near = max(
                0.0,
                self.near_s - near_loiter,
            )

            remain_danger = max(
                0.0,
                self.danger_s - danger_loiter,
            )

    # part 12 screen 11

            if (
                in_near
                and not in_danger
                and track_id not in self._alerted_near
                and near_loiter >= self.near_s
            ):

                self._alerted_near.add(
                    track_id
                )

                events.append(
                    self._event(
                        track,
                        "enter",
                        LEVEL_NEAR,
                        frame_index,
                    )
                )

    # part 13 screen 12

            if (
                in_danger
                and track_id not in self._alerted_danger
                and danger_loiter >= self.danger_s
            ):

                # If the person had a NEAR alert and now
                # entered DANGER, close the NEAR alert first.
                if track_id in self._alerted_near:

                    events.append(
                        self._event(
                            track,
                            "leave",
                            LEVEL_NEAR,
                            frame_index,
                        )
                    )

                    self._alerted_near.discard(
                        track_id
                    )

                self._alerted_danger.add(
                    track_id
                )

                events.append(
                    self._event(
                        track,
                        "enter",
                        LEVEL_DANGER,
                        frame_index,
                    )
                )

    # part 14 screen 13

            if in_danger:

                self._remaining[track_id] = remain_danger

                self._label[track_id] = (
                    f"DANGER: {remain_danger:.1f}s"
                )

                if track_id in self._alerted_danger:

                    self._remaining[track_id] = 0.0
                    self._label[track_id] = "DANGER"

                elif danger_playing:

                    self._remaining[track_id] = (
                        remain_danger
                    )

                    self._label[track_id] = (
                        f"DNG {remain_danger:.1f}"
                    )

                else:

                    self._remaining[track_id] = (
                        self.danger_s
                    )

                    self._label[track_id] = "WALK"

            elif in_near:

                if track_id in self._alerted_near:

                    self._remaining[track_id] = 0.0
                    self._label[track_id] = "NEAR"

                elif near_playing:

                    self._remaining[track_id] = (
                        remain_near
                    )

                    self._label[track_id] = (
                        f"NEAR {remain_near:.1f}"
                    )

                else:

                    self._remaining[track_id] = (
                        self.near_s
                    )

                    self._label[track_id] = "WALK"

        return events

    # part 15 screen 14

    def _tick_missing_danger(
        self,
        present_danger: dict[int, Track],
        frame_index: int,
    ) -> list[RiverbankEvent]:

        events: list[RiverbankEvent] = []

        tracked = (
            set(self._danger_since)
            | set(self._alerted_danger)
        )

        missing = (
            tracked
            - set(present_danger)
        )

        for track_id in list(missing):

            missed = (
                self._grace_danger.get(
                    track_id,
                    0,
                )
                + 1
            )

            if missed < self.grace_frames:

                self._grace_danger[
                    track_id
                ] = missed

                continue

            last = self._last_track.get(
                track_id
            )

            was_alerted = (
                track_id
                in self._alerted_danger
            )

            # Complete danger cleanup.
            self._danger_since.pop(
                track_id,
                None,
            )

            self._grace_danger.pop(
                track_id,
                None,
            )

            self._loiter_danger.pop(
                track_id,
                None,
            )

            self._home_danger.pop(
                track_id,
                None,
            )

            self._alerted_danger.discard(
                track_id
            )

            self._remaining.pop(
                track_id,
                None,
            )

            self._label.pop(
                track_id,
                None,
            )

            if was_alerted:

                events.append(
                    self._leave(
                        track_id,
                        last,
                        LEVEL_DANGER,
                        frame_index,
                    )
                )

        return events

    # part 16 screen 15

    def _tick_missing_near(
        self,
        present_near: dict[int, Track],
        present_danger: dict[int, Track],
        frame_index: int,
    ) -> list[RiverbankEvent]:

        events: list[RiverbankEvent] = []

        tracked = (
            set(self._near_since)
            | set(self._alerted_near)
        )

        missing = (
            tracked
            - set(present_near)
        )

        # If the person is currently in danger,
        # do not treat absence from NEAR as disappearance.
        missing -= set(present_danger)

        for track_id in list(missing):

            missed = (
                self._grace_near.get(
                    track_id,
                    0,
                )
                + 1
            )

            if missed < self.grace_frames:

                self._grace_near[
                    track_id
                ] = missed

                continue

            last = self._last_track.get(
                track_id
            )

            was_alerted = (
                track_id
                in self._alerted_near
            )

            self._drop_near(
                track_id
            )

            if was_alerted:

                events.append(
                    self._leave(
                        track_id,
                        last,
                        LEVEL_NEAR,
                        frame_index,
                    )
                )

            # Completely forget the track if it is
            # no longer active anywhere.
            if (
                track_id
                not in self._danger_since
                and track_id
                not in self._alerted_danger
                and track_id
                not in present_danger
            ):

                self._last_track.pop(
                    track_id,
                    None,
                )

                self._center.pop(
                    track_id,
                    None,
                )

        return events

    # part 16 screen 15
    

    def _drop_near(
        self,
        track_id: int,
    ) -> None:

        self._near_since.pop(
            track_id,
            None,
        )

        self._grace_near.pop(
            track_id,
            None,
        )

        self._loiter_near.pop(
            track_id,
            None,
        )

        self._home_near.pop(
            track_id,
            None,
        )

        self._loiter_near.pop(
            track_id,
            None,
        )

        self._home_near.pop(
            track_id,
            None,
        )

        self._remaining.pop(
            track_id,
            None,
        )

        self._label.pop(
            track_id,
            None,
        )

    # part 17 screen 16

    def _leave(
        self,
        track_id: int,
        last: Track | None,
        level: str,
        frame_index: int,
    ) -> RiverbankEvent:

        if last is not None:

            return self._event(
                last,
                "leave",
                level,
                frame_index,
            )

        return RiverbankEvent(
            track_id=track_id,
            kind="leave",
            level=level,
            frame_index=frame_index,
            class_id=-1,
        )

    def _event(
        self,
        track: Track,
        kind: str,
        level: str,
        frame_index: int,
    ) -> RiverbankEvent:

        return RiverbankEvent(
            track_id=track.track_id,
            kind=kind,
            level=level,
            frame_index=frame_index,
            class_id=track.class_id,
            class_name=track.class_name,
            confidence=track.confidence,
            bbox=track.bbox,
        )

    # part 18 screen 17

    @property
    def near_ids(self) -> set[int]:

        return set(
            self._near_since
        )

    @property
    def danger_ids(self) -> set[int]:

        return set(
            self._danger_since
        )

    @property
    def alerting_near_ids(self) -> set[int]:

        return set(
            self._alerted_near
        )

    @property
    def alerting_danger_ids(self) -> set[int]:

        return set(
            self._alerted_danger
        )



# part 19 screen 18

    def status(
        self,
    ) -> tuple[str, int, int]:

        near_n = len(
            self._near_since
        )

        danger_n = len(
            self._danger_since
        )

        if (
            danger_n > 0
            or self._alerted_danger
        ):
            return (
                LEVEL_DANGER,
                near_n,
                danger_n,
            )

        if (
            near_n > 0
            or self._alerted_near
        ):
            return (
                LEVEL_NEAR,
                near_n,
                danger_n,
            )

        return (
            LEVEL_CLEAR,
            0,
            0,
        )

    # part 20 screen 19

    def draw(
        self,
        frame: np.ndarray,
        *,
        near_alert: bool = False,
        danger_alert: bool = False,
        thickness: int = 2,
        blink_hz: float = 2.0,
    ) -> np.ndarray:

        vis = _draw_level_polygon(
            frame,
            self.near_polygon,
            color=NEAR_COLOR,
            alert=near_alert,
            thickness=thickness,
            blink_hz=blink_hz,
        )

        vis = _draw_level_polygon(
            vis,
            self.danger_polygon,
            color=DANGER_COLOR,
            alert=danger_alert,
            thickness=thickness,
            blink_hz=blink_hz,
        )

        return vis

    def draw_labels(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        vis = frame

        for track_id, text in self._label.items():

            track = self._last_track.get(
                track_id
            )

            if track is None:
                continue

            danger = (
                track_id
                in self._alerted_danger
                or track_id
                in self._danger_since
            )

            color = (
                DANGER_COLOR
                if danger
                else NEAR_COLOR
            )

            xy = _anchor_xy(
                track.bbox,
                self.anchor,
            )

            _draw_anchor_label(
                vis,
                xy,
                text,
                color,
            )

        return vis

    # part 21 screen 20    

    def draw_hud(
        self,
        frame: np.ndarray,
        *,
        height: int = HUD_HEIGHT,
        width: int = HUD_WIDTH,
        blink_hz: float = 2.0,
    ) -> np.ndarray:

        level, near_n, danger_n = self.status()

        if level == LEVEL_DANGER:

            title = "DANGER - ON WATER"
            detail = (
                f"water={danger_n}  "
                f"sidewalk={near_n}"
            )

            color = DANGER_COLOR
            blink = True

            blink_hz = max(
                blink_hz,
                2.5,
            )

        elif level == LEVEL_NEAR:

            title = "NEAR - SIDEWALK"
            detail = (
                f"sidewalk={near_n}"
            )

            color = NEAR_COLOR

            blink = bool(
                self._alerted_near
                or near_n > 0
            )

        else:

            title = "CLEAR"

            detail = (
                "no person in "
                "riverbank zones"
            )

            color = CLEAR_COLOR
            blink = False

        return draw_status_hud(
            frame,
            title=title,
            detail=detail,
            color=color,
            blink=blink,
            height=height,
            width=width,
            blink_hz=blink_hz,
            bright=bool(
                self._alerted_danger
                or self._alerted_near
            ),
        )


    # part 22 screen 21


def draw_status_hud(
    frame: np.ndarray,
    *,
    title: str,
    detail: str,
    color: tuple[int, int, int],
    blink: bool = False,
    height: int = HUD_HEIGHT,
    width: int = HUD_WIDTH,
    blink_hz: float = 2.0,
    bright: bool = False,
) -> np.ndarray:

    if frame.size == 0:
        return frame

    vis = frame

    fh, fw = vis.shape[:2]

    bar_h = max(
        48,
        min(int(height), fh // 3),
    )

    bar_w = max(
        240,
        min(int(width), fw - 24),
    )

    x1 = max(
        0,
        (fw - bar_w) // 2,
    )

    y1 = 8

    x2 = min(
        fw - 1,
        x1 + bar_w,
    )

    y2 = min(
        fh - 1,
        y1 + bar_h,
    )

    alpha = 0.55
    draw_color = color
    border = (255, 255, 255)

    if blink:

        on = (
            time.monotonic()
            * max(0.5, float(blink_hz))
        ) % 1.0 < 0.5

        if on:

            alpha = (
                0.78
                if bright
                else 0.68
            )

        else:

            alpha = 0.28

            draw_color = tuple(
                max(0, c // 3)
                for c in color
            )

            border = tuple(
                max(0, c // 2)
                for c in color
            )

    overlay = vis.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        draw_color,
        -1,
        cv2.LINE_AA,
    )

    vis = cv2.addWeighted(
        overlay,
        alpha,
        vis,
        1.0 - alpha,
        0,
    )

    cv2.rectangle(
        vis,
        (x1, y1),
        (x2, y2),
        border,
        2,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = (
        1.05
        if bar_h >= 90
        else 0.8
    )

    detail_scale = (
        0.7
        if bar_h >= 90
        else 0.55
    )

    title_thick = 3
    detail_thick = 2

    (tw, th), _ = cv2.getTextSize(
        title,
        font,
        title_scale,
        title_thick,
    )

    (dw, dh), _ = cv2.getTextSize(
        detail,
        font,
        detail_scale,
        detail_thick,
    )

    tx = (
        x1
        + max(
            12,
            (bar_w - tw) // 2,
        )
    )

    dx = (
        x1
        + max(
            12,
            (bar_w - dw) // 2,
        )
    )

    ty = y1 + max(
        th + 12,
        bar_h // 2 - 4,
    )

    dy = min(
        y2 - 12,
        ty + dh + 18,
    )

    for text, x, y, scale, thick in (
        (
            title,
            tx,
            ty,
            title_scale,
            title_thick,
        ),
        (
            detail,
            dx,
            dy,
            detail_scale,
            detail_thick,
        ),
    ):

        cv2.putText(
            vis,
            text,
            (x, y),
            font,
            scale,
            (0, 0, 0),
            thick + 2,
            cv2.LINE_AA,
        )

        cv2.putText(
            vis,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

    return vis


    # part 23 screen 22

def _draw_level_polygon(
    frame: np.ndarray,
    polygon: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    alert: bool,
    thickness: int,
    blink_hz: float,
) -> np.ndarray:

    if len(polygon) < 3:
        return frame

    draw_color = color
    draw_alpha = 0.2
    draw_thickness = thickness

    if alert:

        on = (
            time.monotonic()
            * max(0.5, float(blink_hz))
        ) % 1.0 < 0.5

        if on:

            draw_alpha = 0.45
            draw_thickness = max(
                3,
                thickness + 1,
            )

        else:

            draw_color = tuple(
                max(0, c // 2)
                for c in color
            )

            draw_alpha = 0.12

    return draw_zone_polygon(
        frame,
        polygon,
        color=draw_color,
        alpha=draw_alpha,
        thickness=draw_thickness,
    )

    
    # part 24 screen 23

def _anchor_xy(
    bbox: tuple[float, float, float, float],
    anchor: str,
) -> tuple[float, float]:

    x1, y1, x2, y2 = bbox

    cx = (
        x1 + x2
    ) / 2.0

    if anchor == "bottom":
        return cx, y2

    if anchor == "top":
        return cx, y1

    return (
        cx,
        (y1 + y2) / 2.0,
    )

    
    # part 25 screen 24
    
def _draw_anchor_label(
    frame: np.ndarray,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
) -> None:

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.5
    thick = 1

    (tw, th), base = cv2.getTextSize(
        text,
        font,
        scale,
        thick,
    )

    x0 = (
        int(xy[0])
        - tw // 2
        - 4
    )

    y0 = (
        int(xy[1])
        + 2
    )

    x1 = (
        x0
        + tw
        + 8
    )

    y1 = (
        y0
        + th
        + base
        + 6
    )

    cv2.rectangle(
        frame,
        (x0, y0),
        (x1, y1),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (
            x0 + 4,
            y0 + th + 2,
        ),
        font,
        scale,
        color,
        thick,
        cv2.LINE_AA,
    )


    # last part 26 screen 25

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO)

    monitor = RiverbankMonitor(
            near_polygon=[(0, 300), (640, 300), (640, 400), (0, 400)],
            danger_polygon=[(0, 100), (640, 100), (640, 300), (0, 300)],
            fps=10.0,
        )

    for i in range(45):
        child = Track(
            track_id=1,
            bbox=(100, 300, 140, 380),
            class_name="person",
        )

        for ev in monitor.update([child]):
            print(
                f"frame={i}: {ev.kind} {ev.level} id={ev.track_id}"
            )





