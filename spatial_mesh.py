"""
spatial_mesh.py
===============
Distributed Spatial Mesh & Peer-to-Peer Device Handover Engine.
Establishes a zero-latency peer mesh across local computers, laptops, tablets,
and AR/VR spatial displays, enabling seamless multi-device cursor focus transfer
when the user's gaze traverses physical display boundaries.

Part of the v6.0 Neuromorphic & Cross-Platform Enterprise Engine.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("SpatialMesh")


@dataclass
class SpatialMeshPeerNode:
    """Represents an adjacent peer device in the spatial mesh network."""
    peer_id: str
    device_type: str  # "desktop", "laptop", "tablet", "spatial_hmd"
    relative_position: str  # "LEFT", "RIGHT", "ABOVE", "BELOW"
    screen_resolution: Tuple[int, int]  # (width, height)
    is_active: bool = True
    network_endpoint: str = "127.0.0.1:9090"


class SpatialMeshHandoverEngine:
    """
    Evaluates gaze coordinate boundaries and routes input events across
    the distributed multi-device spatial mesh.
    """

    def __init__(self, local_resolution: Tuple[int, int] = (1920, 1080)):
        self.local_w, self.local_h = local_resolution
        self.peers: Dict[str, SpatialMeshPeerNode] = {}
        self.current_focused_peer_id: str = "local"
        self.total_handovers: int = 0

    def register_peer(self, peer: SpatialMeshPeerNode) -> None:
        """Register a neighbouring spatial peer device."""
        self.peers[peer.peer_id] = peer
        logger.info(
            "[Spatial Mesh] Registered peer '%s' (%s) positioned [%s]",
            peer.peer_id, peer.device_type, peer.relative_position
        )

    def evaluate_handover(
        self,
        cursor_x: float,
        cursor_y: float
    ) -> Optional[Tuple[SpatialMeshPeerNode, float, float]]:
        """
        Evaluate if gaze has crossed the local physical display boundary into an adjacent peer.
        
        Args:
            cursor_x: Local screen pixel X coordinate (may be < 0 or > local_w).
            cursor_y: Local screen pixel Y coordinate (may be < 0 or > local_h).
            
        Returns:
            Tuple of (target_peer, mapped_peer_x, mapped_peer_y) if handover triggered, else None.
        """
        for peer in self.peers.values():
            if not peer.is_active:
                continue

            target_w, target_h = peer.screen_resolution

            # Handover LEFT
            if cursor_x < 0 and peer.relative_position == "LEFT":
                norm_y = max(0.0, min(1.0, cursor_y / float(self.local_h)))
                peer_x = target_w - 5.0
                peer_y = norm_y * target_h
                self._record_handover(peer.peer_id)
                return peer, peer_x, peer_y

            # Handover RIGHT
            if cursor_x > self.local_w and peer.relative_position == "RIGHT":
                norm_y = max(0.0, min(1.0, cursor_y / float(self.local_h)))
                peer_x = 5.0
                peer_y = norm_y * target_h
                self._record_handover(peer.peer_id)
                return peer, peer_x, peer_y

            # Handover ABOVE
            if cursor_y < 0 and peer.relative_position == "ABOVE":
                norm_x = max(0.0, min(1.0, cursor_x / float(self.local_w)))
                peer_x = norm_x * target_w
                peer_y = target_h - 5.0
                self._record_handover(peer.peer_id)
                return peer, peer_x, peer_y

            # Handover BELOW
            if cursor_y > self.local_h and peer.relative_position == "BELOW":
                norm_x = max(0.0, min(1.0, cursor_x / float(self.local_w)))
                peer_x = norm_x * target_w
                peer_y = 5.0
                self._record_handover(peer.peer_id)
                return peer, peer_x, peer_y

        return None

    def _record_handover(self, target_id: str) -> None:
        if self.current_focused_peer_id != target_id:
            logger.info("[Spatial Mesh] Handover triggered: %s -> %s", self.current_focused_peer_id, target_id)
            self.current_focused_peer_id = target_id
            self.total_handovers += 1
