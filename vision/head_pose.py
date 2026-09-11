"""
Lightweight facial-landmark based head-pose estimation.

This module:
- accepts MediaPipe facial landmarks
- estimates yaw, pitch and roll
- returns a structured result
- performs no video capture, Android communication or GUI operations
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


# MediaPipe Face Mesh landmark indices.
# These points form a stable approximate head coordinate frame.
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291


@dataclass(frozen=True)
class HeadPoseResult:
    """Estimated head orientation."""

    yaw: float
    pitch: float
    roll: float
    valid: bool
    confidence: float


class HeadPoseEstimator:
    """
    Lightweight head-pose estimator using MediaPipe landmarks.

    Coordinate convention:
        yaw:
            Negative = head turned to the user's left.
            Positive = head turned to the user's right.

        pitch:
            Negative = looking/head tilted downward.
            Positive = looking/head tilted upward.

        roll:
            Negative = head tilted toward the user's left shoulder.
            Positive = head tilted toward the user's right shoulder.

    Angles are returned in degrees.
    """

    def __init__(
        self,
        max_yaw: float = 100.0,
        max_pitch: float = 80.0,
        max_roll: float = 80.0,
    ) -> None:
        self.max_yaw = float(max_yaw)
        self.max_pitch = float(max_pitch)
        self.max_roll = float(max_roll)

    @staticmethod
    def _point(
        landmarks: Sequence,
        index: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Safely extract one MediaPipe landmark."""
        if landmarks is None:
            return None

        try:
            landmark = landmarks[index]

            x = float(landmark.x)
            y = float(landmark.y)
            z = float(getattr(landmark, "z", 0.0))

        except (IndexError, AttributeError, TypeError, ValueError):
            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and math.isfinite(z)
        ):
            return None

        return x, y, z

    @staticmethod
    def _distance(
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
    ) -> float:
        """3-D Euclidean distance between two landmarks."""
        return math.sqrt(
            (a[0] - b[0]) ** 2
            + (a[1] - b[1]) ** 2
            + (a[2] - b[2]) ** 2
        )

    def estimate(self, landmarks: Sequence) -> HeadPoseResult:
        """
        Estimate head pose from MediaPipe Face Mesh landmarks.

        The calculation uses:
        - both outer eye corners for the horizontal head axis
        - both mouth corners for a second horizontal reference
        - nose position relative to the face center
        - MediaPipe's normalized landmark Z coordinate

        No image/frame processing is performed here.
        """

        left_eye = self._point(landmarks, LEFT_EYE_OUTER)
        right_eye = self._point(landmarks, RIGHT_EYE_OUTER)
        nose = self._point(landmarks, NOSE_TIP)
        left_mouth = self._point(landmarks, LEFT_MOUTH_CORNER)
        right_mouth = self._point(landmarks, RIGHT_MOUTH_CORNER)

        required_points = (
            left_eye,
            right_eye,
            nose,
            left_mouth,
            right_mouth,
        )

        if any(point is None for point in required_points):
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                valid=False,
                confidence=0.0,
            )

        # Type narrowing for static type checkers.
        assert left_eye is not None
        assert right_eye is not None
        assert nose is not None
        assert left_mouth is not None
        assert right_mouth is not None

        # ----------------------------------------------------------
        # 1. Face horizontal axis
        #
        # The line between the outer eye corners is approximately
        # horizontal when the head is facing the camera.
        # ----------------------------------------------------------
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]

        eye_distance = math.hypot(eye_dx, eye_dy)

        if eye_distance < 1e-6:
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                valid=False,
                confidence=0.0,
            )

        # ----------------------------------------------------------
        # 2. Roll
        #
        # atan2() of the eye-line gives the in-plane rotation.
        # ----------------------------------------------------------
        roll = math.degrees(
            math.atan2(eye_dy, eye_dx)
        )

        # ----------------------------------------------------------
        # 3. Face center
        #
        # Average the eye and mouth pairs to reduce sensitivity
        # to any one noisy landmark.
        # ----------------------------------------------------------
        eye_center_x = (
            left_eye[0] + right_eye[0]
        ) * 0.5

        eye_center_y = (
            left_eye[1] + right_eye[1]
        ) * 0.5

        eye_center_z = (
            left_eye[2] + right_eye[2]
        ) * 0.5

        mouth_center_x = (
            left_mouth[0] + right_mouth[0]
        ) * 0.5

        mouth_center_y = (
            left_mouth[1] + right_mouth[1]
        ) * 0.5

        mouth_center_z = (
            left_mouth[2] + right_mouth[2]
        ) * 0.5

        face_center = (
            (eye_center_x + mouth_center_x) * 0.5,
            (eye_center_y + mouth_center_y) * 0.5,
            (eye_center_z + mouth_center_z) * 0.5,
        )

        # ----------------------------------------------------------
        # 4. Yaw
        #
        # When the head rotates horizontally, the nose moves
        # relative to the midpoint of the eye/mouth structure.
        #
        # Normalize by face width so the estimate is less dependent
        # on how close the face is to the camera.
        # ----------------------------------------------------------
        face_width = self._distance(
            left_eye,
            right_eye,
        )

        if face_width < 1e-6:
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=0.0,
                valid=False,
                confidence=0.0,
            )

        normalized_nose_x = (
            nose[0] - face_center[0]
        ) / face_width

        # The scale factor converts the normalized horizontal
        # displacement into a practical angle estimate.
        yaw = math.degrees(
            math.atan(normalized_nose_x * 2.0)
        )

        # ----------------------------------------------------------
        # 5. Pitch
        #
        # Use the vertical relationship between the nose, eye center,
        # and mouth center. This gives a lightweight approximation
        # of vertical head rotation.
        #
        # The nose's MediaPipe Z coordinate provides additional
        # depth information, which helps distinguish some forward/
        # backward head motion.
        # ----------------------------------------------------------
        eye_to_mouth_y = (
            mouth_center_y - eye_center_y
        )

        if abs(eye_to_mouth_y) < 1e-6:
            return HeadPoseResult(
                yaw=0.0,
                pitch=0.0,
                roll=float(roll),
                valid=False,
                confidence=0.0,
            )

        nose_vertical_ratio = (
            nose[1] - eye_center_y
        ) / abs(eye_to_mouth_y)

        pitch = math.degrees(
            math.atan(
                (nose_vertical_ratio - 0.45) * 2.0
            )
        )

        # MediaPipe Y grows downward. The convention above therefore
        # naturally produces a sign opposite to a mathematical
        # upward-positive image coordinate system. Flip it so:
        # positive = head tilted upward.
        pitch = -pitch

        # ----------------------------------------------------------
        # 6. Plausibility / validity
        # ----------------------------------------------------------
        valid = (
            abs(yaw) <= self.max_yaw
            and abs(pitch) <= self.max_pitch
            and abs(roll) <= self.max_roll
        )

        if not valid:
            return HeadPoseResult(
                yaw=float(yaw),
                pitch=float(pitch),
                roll=float(roll),
                valid=False,
                confidence=0.0,
            )

        # ----------------------------------------------------------
        # 7. Lightweight confidence estimate
        #
        # Confidence is based on the stability of the geometric
        # reference distances. It is NOT a MediaPipe confidence score.
        # ----------------------------------------------------------
        eye_width = self._distance(
            left_eye,
            right_eye,
        )

        mouth_width = self._distance(
            left_mouth,
            right_mouth,
        )

        if eye_width <= 1e-6 or mouth_width <= 1e-6:
            confidence = 0.0
        else:
            width_ratio = mouth_width / eye_width

            # Typical facial geometry keeps these two widths within
            # a broad reasonable range. Avoid making the validity
            # gate too aggressive.
            geometry_quality = max(
                0.0,
                min(
                    1.0,
                    1.0 - abs(width_ratio - 0.85) / 1.0,
                ),
            )

            confidence = float(
                0.5 + 0.5 * geometry_quality
            )

        return HeadPoseResult(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            valid=True,
            confidence=confidence,
        )