"""
MediaPipe Face Mesh / Iris based eye-geometry tracking.

This module:
- identifies left/right eye landmarks
- identifies iris landmarks
- calculates iris centers
- calculates eye corners
- calculates normalized iris X/Y
- calculates six-point Eye Aspect Ratio (EAR)
- performs no image thresholding or face-box cropping

Coordinate convention:
- MediaPipe landmark x/y values are normalized to approximately [0, 1].
- x increases from left to right in the image.
- y increases from top to bottom in the image.
- z is MediaPipe's relative depth coordinate.
- Normalized iris coordinates are expressed relative to each eye:
    X = 0.0 -> left eye corner
    X = 1.0 -> right eye corner
    Y = 0.0 -> upper eye boundary
    Y = 1.0 -> lower eye boundary
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices
# ---------------------------------------------------------------------------

# Left eye.
# MediaPipe's "left" refers to the person's left eye, which appears
# on the right side of a non-mirrored camera image.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133

# Six-point EAR geometry for the left eye.
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145

# Additional vertical pair used for the standard six-point EAR.
LEFT_EYE_UPPER_2 = 160
LEFT_EYE_LOWER_2 = 144


# Right eye.
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362

RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374

RIGHT_EYE_UPPER_2 = 385
RIGHT_EYE_LOWER_2 = 380


# MediaPipe Iris landmark indices.
#
# Face Mesh with refine_landmarks=True provides 5 iris landmarks
# for each eye.
LEFT_IRIS_LANDMARKS = (
    468,
    469,
    470,
    471,
    472,
)

RIGHT_IRIS_LANDMARKS = (
    473,
    474,
    475,
    476,
    477,
)


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EyeGeometry:
    """Geometry calculated for one eye."""

    iris_center: Optional[Tuple[float, float]]
    eye_outer_corner: Optional[Tuple[float, float]]
    eye_inner_corner: Optional[Tuple[float, float]]

    normalized_iris_x: Optional[float]
    normalized_iris_y: Optional[float]

    ear: Optional[float]
    valid: bool


@dataclass(frozen=True)
class IrisTrackingResult:
    """Complete left/right iris and eye-geometry result."""

    left: EyeGeometry
    right: EyeGeometry

    valid: bool


# ---------------------------------------------------------------------------
# IrisTracker
# ---------------------------------------------------------------------------


class IrisTracker:
    """
    Lightweight MediaPipe Face Mesh / Iris eye-geometry tracker.

    The class accepts an already-computed MediaPipe landmark list.

    It does NOT:
    - capture video
    - detect faces
    - crop face/eye boxes
    - threshold pixels
    - detect pupils using OpenCV
    - communicate with Android
    - create GUI windows
    """

    def __init__(
        self,
        min_normalized_value: float = 0.0,
        max_normalized_value: float = 1.0,
    ) -> None:
        self.min_normalized_value = float(min_normalized_value)
        self.max_normalized_value = float(max_normalized_value)

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_point(
        landmarks: Sequence,
        index: int,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Safely return a MediaPipe landmark as (x, y, z).

        Invalid, missing or non-finite landmarks return None.
        """

        if landmarks is None:
            return None

        try:
            landmark = landmarks[index]

            x = float(landmark.x)
            y = float(landmark.y)
            z = float(getattr(landmark, "z", 0.0))

        except (
            IndexError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(z)
        ):
            return None

        return x, y, z

    def _get_points(
        self,
        landmarks: Sequence,
        indices: Sequence[int],
    ) -> Optional[
        Tuple[Tuple[float, float, float], ...]
    ]:
        """Safely extract several landmarks."""

        points = []

        for index in indices:
            point = self._get_point(landmarks, index)

            if point is None:
                return None

            points.append(point)

        return tuple(points)

    # ------------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _distance(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """Calculate 3-D Euclidean distance."""

        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )

    @staticmethod
    def _distance_2d(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """Calculate 2-D image-space distance."""

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    @staticmethod
    def _center(
        points: Sequence[
            Tuple[float, float, float]
        ],
    ) -> Tuple[float, float]:
        """Calculate the 2-D center of a group of landmarks."""

        count = len(points)

        return (
            sum(point[0] for point in points) / count,
            sum(point[1] for point in points) / count,
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """Clamp a value to a valid normalized range."""

        return max(
            minimum,
            min(maximum, value),
        )

    # ------------------------------------------------------------------
    # Iris normalization
    # ------------------------------------------------------------------

    def _normalized_iris_position(
        self,
        iris_center: Tuple[float, float],
        outer_corner: Tuple[float, float, float],
        inner_corner: Tuple[float, float, float],
        upper_points: Sequence[
            Tuple[float, float, float]
        ],
        lower_points: Sequence[
            Tuple[float, float, float]
        ],
    ) -> Optional[Tuple[float, float]]:
        """
        Convert iris position into coordinates relative to the eye.

        X:
            Projection onto the line between the two eye corners.

            0.0 = outer corner
            1.0 = inner corner

        Y:
            Relative position between the upper and lower eye
            boundaries.

            0.0 = upper boundary
            1.0 = lower boundary

        Projection is used instead of simply dividing raw x/y
        coordinates, making the result more stable when the head
        is rotated in the image.
        """

        eye_dx = inner_corner[0] - outer_corner[0]
        eye_dy = inner_corner[1] - outer_corner[1]

        eye_width_squared = (
            eye_dx * eye_dx
            + eye_dy * eye_dy
        )

        if eye_width_squared < 1e-10:
            return None

        iris_dx = iris_center[0] - outer_corner[0]
        iris_dy = iris_center[1] - outer_corner[1]

        # Projection of iris center onto the eye-corner axis.
        normalized_x = (
            iris_dx * eye_dx
            + iris_dy * eye_dy
        ) / eye_width_squared

        upper_center = self._center(upper_points)
        lower_center = self._center(lower_points)

        vertical_dx = (
            lower_center[0] - upper_center[0]
        )

        vertical_dy = (
            lower_center[1] - upper_center[1]
        )

        vertical_length_squared = (
            vertical_dx * vertical_dx
            + vertical_dy * vertical_dy
        )

        if vertical_length_squared < 1e-10:
            return None

        iris_vertical_dx = (
            iris_center[0] - upper_center[0]
        )

        iris_vertical_dy = (
            iris_center[1] - upper_center[1]
        )

        normalized_y = (
            iris_vertical_dx * vertical_dx
            + iris_vertical_dy * vertical_dy
        ) / vertical_length_squared

        normalized_x = self._clamp(
            normalized_x,
            self.min_normalized_value,
            self.max_normalized_value,
        )

        normalized_y = self._clamp(
            normalized_y,
            self.min_normalized_value,
            self.max_normalized_value,
        )

        return normalized_x, normalized_y

    # ------------------------------------------------------------------
    # Six-point EAR
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_ear(
        eye_outer: Tuple[float, float, float],
        eye_inner: Tuple[float, float, float],
        upper_1: Tuple[float, float, float],
        lower_1: Tuple[float, float, float],
        upper_2: Tuple[float, float, float],
        lower_2: Tuple[float, float, float],
    ) -> Optional[float]:
        """
        Calculate standard six-point Eye Aspect Ratio.

        EAR = (
            |p2-p6| + |p3-p5|
        ) / (
            2 * |p1-p4|
        )

        Here:
            p1 = outer eye corner
            p4 = inner eye corner
            p2/p3 = upper eyelid points
            p5/p6 = corresponding lower eyelid points

        A larger EAR means a more open eye.
        A smaller EAR means a more closed eye.
        """

        horizontal = IrisTracker._distance_2d(
            eye_outer,
            eye_inner,
        )

        if horizontal < 1e-8:
            return None

        vertical_1 = IrisTracker._distance_2d(
            upper_1,
            lower_1,
        )

        vertical_2 = IrisTracker._distance_2d(
            upper_2,
            lower_2,
        )

        ear = (
            vertical_1 + vertical_2
        ) / (
            2.0 * horizontal
        )

        if not math.isfinite(ear):
            return None

        return float(ear)

    # ------------------------------------------------------------------
    # Per-eye processing
    # ------------------------------------------------------------------

    def _process_eye(
        self,
        landmarks: Sequence,
        outer_index: int,
        inner_index: int,
        upper_index_1: int,
        lower_index_1: int,
        upper_index_2: int,
        lower_index_2: int,
        iris_indices: Sequence[int],
    ) -> EyeGeometry:
        """Calculate geometry for one eye."""

        outer = self._get_point(
            landmarks,
            outer_index,
        )

        inner = self._get_point(
            landmarks,
            inner_index,
        )

        upper_1 = self._get_point(
            landmarks,
            upper_index_1,
        )

        lower_1 = self._get_point(
            landmarks,
            lower_index_1,
        )

        upper_2 = self._get_point(
            landmarks,
            upper_index_2,
        )

        lower_2 = self._get_point(
            landmarks,
            lower_index_2,
        )

        iris_points = self._get_points(
            landmarks,
            iris_indices,
        )

        # EAR can be calculated even if iris landmarks are unavailable.
        ear = None

        if all(
            point is not None
            for point in (
                outer,
                inner,
                upper_1,
                lower_1,
                upper_2,
                lower_2,
            )
        ):
            assert outer is not None
            assert inner is not None
            assert upper_1 is not None
            assert lower_1 is not None
            assert upper_2 is not None
            assert lower_2 is not None

            ear = self._calculate_ear(
                outer,
                inner,
                upper_1,
                lower_1,
                upper_2,
                lower_2,
            )

        # Iris geometry requires all five iris landmarks.
        if (
            outer is None
            or inner is None
            or iris_points is None
        ):
            return EyeGeometry(
                iris_center=None,
                eye_outer_corner=(
                    (outer[0], outer[1])
                    if outer is not None
                    else None
                ),
                eye_inner_corner=(
                    (inner[0], inner[1])
                    if inner is not None
                    else None
                ),
                normalized_iris_x=None,
                normalized_iris_y=None,
                ear=ear,
                valid=False,
            )

        iris_center = self._center(
            iris_points
        )

        upper_points = []

        if upper_1 is not None:
            upper_points.append(upper_1)

        if upper_2 is not None:
            upper_points.append(upper_2)

        lower_points = []

        if lower_1 is not None:
            lower_points.append(lower_1)

        if lower_2 is not None:
            lower_points.append(lower_2)

        if not upper_points or not lower_points:
            return EyeGeometry(
                iris_center=iris_center,
                eye_outer_corner=(
                    outer[0],
                    outer[1],
                ),
                eye_inner_corner=(
                    inner[0],
                    inner[1],
                ),
                normalized_iris_x=None,
                normalized_iris_y=None,
                ear=ear,
                valid=False,
            )

        normalized = self._normalized_iris_position(
            iris_center,
            outer,
            inner,
            upper_points,
            lower_points,
        )

        if normalized is None:
            return EyeGeometry(
                iris_center=iris_center,
                eye_outer_corner=(
                    outer[0],
                    outer[1],
                ),
                eye_inner_corner=(
                    inner[0],
                    inner[1],
                ),
                normalized_iris_x=None,
                normalized_iris_y=None,
                ear=ear,
                valid=False,
            )

        normalized_x, normalized_y = normalized

        return EyeGeometry(
            iris_center=iris_center,
            eye_outer_corner=(
                outer[0],
                outer[1],
            ),
            eye_inner_corner=(
                inner[0],
                inner[1],
            ),
            normalized_iris_x=normalized_x,
            normalized_iris_y=normalized_y,
            ear=ear,
            valid=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        landmarks: Sequence,
    ) -> IrisTrackingResult:
        """
        Process one MediaPipe landmark list.

        Args:
            landmarks:
                MediaPipe Face Mesh landmarks, normally obtained from
                `results.multi_face_landmarks[0].landmark`.

        Returns:
            IrisTrackingResult containing left/right eye geometry.
        """

        if landmarks is None:
            invalid_eye = EyeGeometry(
                iris_center=None,
                eye_outer_corner=None,
                eye_inner_corner=None,
                normalized_iris_x=None,
                normalized_iris_y=None,
                ear=None,
                valid=False,
            )

            return IrisTrackingResult(
                left=invalid_eye,
                right=invalid_eye,
                valid=False,
            )

        left = self._process_eye(
            landmarks=landmarks,
            outer_index=LEFT_EYE_OUTER,
            inner_index=LEFT_EYE_INNER,
            upper_index_1=LEFT_EYE_UPPER,
            lower_index_1=LEFT_EYE_LOWER,
            upper_index_2=LEFT_EYE_UPPER_2,
            lower_index_2=LEFT_EYE_LOWER_2,
            iris_indices=LEFT_IRIS_LANDMARKS,
        )

        right = self._process_eye(
            landmarks=landmarks,
            outer_index=RIGHT_EYE_OUTER,
            inner_index=RIGHT_EYE_INNER,
            upper_index_1=RIGHT_EYE_UPPER,
            lower_index_1=RIGHT_EYE_LOWER,
            upper_index_2=RIGHT_EYE_UPPER_2,
            lower_index_2=RIGHT_EYE_LOWER_2,
            iris_indices=RIGHT_IRIS_LANDMARKS,
        )

        return IrisTrackingResult(
            left=left,
            right=right,
            valid=(
                left.valid
                or right.valid
            ),
        )

    # Alias useful when integrating with a vision pipeline.
    update = process