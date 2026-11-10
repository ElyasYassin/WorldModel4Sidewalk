"""Known-map global planning: occupancy grid A* -> sparse waypoints + progress tracking."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ..sim.base import SceneSpec


@dataclass
class OccupancyGrid:
    origin: np.ndarray  # world xy of cell (0, 0) center
    resolution: float
    occupied: np.ndarray  # [nx, ny] bool

    @classmethod
    def from_scene(cls, scene: SceneSpec, resolution: float = 0.1, inflation: float = 0.45) -> "OccupancyGrid":
        xmin, xmax, ymin, ymax = scene.bounds
        xs = np.arange(xmin + resolution / 2, xmax, resolution)
        ys = np.arange(ymin + resolution / 2, ymax, resolution)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        occ = scene.clearance(np.stack([X, Y], -1), robot_radius=inflation) < 0
        return cls(np.array([xs[0], ys[0]]), resolution, occ)

    def to_cell(self, xy) -> tuple[int, int]:
        ij = np.round((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        ij = np.clip(ij, 0, np.array(self.occupied.shape) - 1)
        return int(ij[0]), int(ij[1])

    def to_world(self, ij) -> np.ndarray:
        return self.origin + np.asarray(ij, dtype=np.float64) * self.resolution

    def nearest_free(self, ij: tuple[int, int]) -> tuple[int, int]:
        if not self.occupied[ij]:
            return ij
        free = np.argwhere(~self.occupied)
        if len(free) == 0:
            raise RuntimeError("no free cells in map")
        k = np.argmin(((free - np.array(ij)) ** 2).sum(1))
        return int(free[k, 0]), int(free[k, 1])


def astar(grid: OccupancyGrid, start_xy, goal_xy) -> np.ndarray | None:
    """8-connected A*. Returns [N, 2] world points or None if unreachable."""
    s = grid.nearest_free(grid.to_cell(start_xy))
    g = grid.nearest_free(grid.to_cell(goal_xy))
    nx, ny = grid.occupied.shape
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 2**0.5), (1, -1, 2**0.5), (-1, 1, 2**0.5), (-1, -1, 2**0.5)]

    def h(c):
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (dx + dy) + (2**0.5 - 2) * min(dx, dy)

    openq = [(h(s), 0.0, s)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {s: 0.0}
    closed = set()
    while openq:
        _, c, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return grid.to_world(np.array(path[::-1]))
        closed.add(cur)
        for dx, dy, w in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny) or grid.occupied[nb]:
                continue
            if dx and dy and (grid.occupied[cur[0] + dx, cur[1]] or grid.occupied[cur[0], cur[1] + dy]):
                continue  # no corner cutting
            nc = c + w
            if nc < cost.get(nb, np.inf):
                cost[nb] = nc
                came[nb] = cur
                heapq.heappush(openq, (nc + h(nb), nc, nb))
    return None


def sparsify(path: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a dense path to waypoints every `spacing` meters (always ends at the goal)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return path[-1:]
    targets = np.arange(spacing, s[-1], spacing)
    pts = np.stack([np.interp(targets, s, path[:, 0]), np.interp(targets, s, path[:, 1])], -1) if len(targets) else np.zeros((0, 2))
    return np.vstack([pts, path[-1:]])


class GlobalPlanner:
    def __init__(self, resolution: float = 0.1, inflation: float = 0.45, spacing: float = 1.0):
        self.resolution, self.inflation, self.spacing = resolution, inflation, spacing
        self.grid: OccupancyGrid | None = None

    def plan(self, scene: SceneSpec, start_xy, goal_xy) -> np.ndarray:
        if self.grid is None or getattr(self, "_scene_id", None) != scene.scene_id:
            self.grid = OccupancyGrid.from_scene(scene, self.resolution, self.inflation)
            self._scene_id = scene.scene_id
        path = astar(self.grid, start_xy, goal_xy)
        if path is None:
            raise RuntimeError(f"no global path in scene {scene.scene_id}")
        return sparsify(path, self.spacing)


class WaypointTracker:
    """Advances along waypoints; the final waypoint is the goal."""

    def __init__(self, waypoints: np.ndarray, reach_tol: float = 0.6):
        self.wps = np.asarray(waypoints, dtype=np.float64)
        self.idx = 0
        self.reach_tol = reach_tol

    def update(self, xy) -> np.ndarray:
        xy = np.asarray(xy)
        while self.idx < len(self.wps) - 1:
            d = np.linalg.norm(self.wps[self.idx] - xy)
            d_next = np.linalg.norm(self.wps[self.idx + 1] - xy)
            if d < self.reach_tol or d_next < d - 0.5:  # reached or clearly passed
                self.idx += 1
            else:
                break
        return self.wps[self.idx]

    @property
    def final(self) -> bool:
        return self.idx == len(self.wps) - 1
