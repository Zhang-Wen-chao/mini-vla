"""Minimal headless PushT simulator used only for Phase 1 rollout metrics.

The physics, geometry and coverage definition are adapted from the MIT-licensed
PushT environment in `real-stanford/diffusion_policy`.  It intentionally has
no Gym dependency: the training package keeps its torch/numpy/zarr-only core.
Install ``mini-vla[rollout]`` to use this module.
"""

from __future__ import annotations

import os

# Must be set before pygame imports so L20 can render without an X display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pymunk
from pymunk.vec2d import Vec2d
import shapely.geometry as sg


def _body_geometry(body: pymunk.Body, shapes: list[pymunk.Shape]) -> sg.MultiPolygon:
    polygons = []
    for shape in shapes:
        if not isinstance(shape, pymunk.Poly):
            raise TypeError(f"PushT expects polygon shapes, got {type(shape)}")
        vertices = [body.local_to_world(vertex) for vertex in shape.get_vertices()]
        polygons.append(sg.Polygon([(point.x, point.y) for point in vertices]))
    return sg.MultiPolygon(polygons)


class PushTImageEnv:
    """Push a T-shaped block into the target pose with absolute XY actions.

    ``step(action)`` advances 0.1 s of the official 100 Hz simulation with a
    10 Hz PD controller.  It returns an RGB image in HWC uint8 format, the
    current coverage, terminal success, and diagnostics.
    """

    window_size = 512
    sim_hz = 100
    control_hz = 10
    success_threshold = 0.95

    def __init__(self, render_size: int = 96, legacy: bool = False) -> None:
        self.render_size = render_size
        self.legacy = legacy
        self._seed = 0
        self.k_p, self.k_v = 100, 20
        self.space: pymunk.Space | None = None
        self.agent: pymunk.Body | None = None
        self.block: pymunk.Body | None = None
        self.goal_pose = np.array([256.0, 256.0, np.pi / 4])
        self.goal_color = pygame.Color("LightGreen")
        self.n_contact_points = 0

    def seed(self, seed: int) -> None:
        self._seed = seed

    def reset(self, seed: int | None = None, state: np.ndarray | None = None) -> np.ndarray:
        """Reset from the official seeded distribution or an explicit five-value state.

        The optional ``state`` is intentionally exposed for evaluator calibration
        only.  Its layout is ``(agent_x, agent_y, block_x, block_y, block_angle)``
        and matches the replay zarr's ``data/state`` field.
        """
        if seed is not None:
            self.seed(seed)
        self._setup()
        if state is None:
            rng = np.random.RandomState(self._seed)
            state = np.array([
                rng.randint(50, 450), rng.randint(50, 450),
                rng.randint(100, 400), rng.randint(100, 400),
                rng.randn() * 2 * np.pi - np.pi,
            ])
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (5,):
            raise ValueError("state must have shape (5,)")
        self._set_state(state)
        return self.render()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.space is None or self.agent is None or self.block is None:
            raise RuntimeError("call reset before step")
        action = np.clip(np.asarray(action, dtype=np.float64), 0, self.window_size)
        if action.shape != (2,):
            raise ValueError("action must have shape (2,)")
        target = Vec2d(float(action[0]), float(action[1]))
        self.n_contact_points = 0
        dt = 1.0 / self.sim_hz
        for _ in range(self.sim_hz // self.control_hz):
            acceleration = self.k_p * (target - self.agent.position) - self.k_v * self.agent.velocity
            self.agent.velocity += acceleration * dt
            self.space.step(dt)
        coverage = self.coverage()
        done = coverage > self.success_threshold
        info = {
            "coverage": coverage,
            "pos_agent": np.array(self.agent.position),
            "block_pose": np.array([*self.block.position, self.block.angle]),
            "goal_pose": self.goal_pose.copy(),
        }
        return self.render(), min(coverage / self.success_threshold, 1.0), done, info

    def coverage(self) -> float:
        if self.block is None:
            raise RuntimeError("call reset before coverage")
        goal_body = pymunk.Body(1, pymunk.moment_for_box(1, (50, 100)))
        goal_body.position = tuple(self.goal_pose[:2])
        goal_body.angle = float(self.goal_pose[2])
        goal = _body_geometry(goal_body, self.block.shapes)
        block = _body_geometry(self.block, self.block.shapes)
        return float(goal.intersection(block).area / goal.area)

    def agent_position(self) -> np.ndarray:
        """Current agent XY in the 512x512 workspace."""
        if self.agent is None:
            raise RuntimeError("call reset before agent_position")
        return np.asarray(self.agent.position, dtype=np.float32)

    def render(self) -> np.ndarray:
        if self.space is None or self.block is None:
            raise RuntimeError("call reset before render")
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self._draw_polygon(canvas, self._goal_body(), self.block.shapes, self.goal_color)
        for wall in self.walls:
            pygame.draw.line(canvas, pygame.Color("LightGray"), wall.a, wall.b, width=4)
        self._draw_circle(canvas, self.agent, 15, pygame.Color("RoyalBlue"))
        self._draw_polygon(canvas, self.block, self.block.shapes, pygame.Color("LightSlateGray"))
        if self.render_size != self.window_size:
            canvas = pygame.transform.smoothscale(canvas, (self.render_size, self.render_size))
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2)).copy()

    def close(self) -> None:
        pygame.quit()

    def _setup(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0
        self.walls = [
            self._add_wall((5, 506), (5, 5)), self._add_wall((5, 5), (506, 5)),
            self._add_wall((506, 5), (506, 506)), self._add_wall((5, 506), (506, 506)),
        ]
        self.agent = self._add_circle((256, 400), 15)
        self.block = self._add_tee((256, 300), 0)
        handler = self.space.add_collision_handler(0, 0)
        handler.post_solve = self._handle_collision

    def _add_wall(self, start: tuple[int, int], end: tuple[int, int]) -> pymunk.Segment:
        assert self.space is not None
        wall = pymunk.Segment(self.space.static_body, start, end, 2)
        self.space.add(wall)
        return wall

    def _add_circle(self, position: tuple[int, int], radius: int) -> pymunk.Body:
        assert self.space is not None
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        shape = pymunk.Circle(body, radius)
        shape.friction = 1
        self.space.add(body, shape)
        return body

    def _add_tee(self, position: tuple[int, int], angle: float, scale: int = 30) -> pymunk.Body:
        assert self.space is not None
        mass, length = 1, 4
        vertices1 = [(-length * scale / 2, scale), (length * scale / 2, scale),
                     (length * scale / 2, 0), (-length * scale / 2, 0)]
        vertices2 = [(-scale / 2, scale), (-scale / 2, length * scale),
                     (scale / 2, length * scale), (scale / 2, scale)]
        inertia = pymunk.moment_for_poly(mass, vertices1) + pymunk.moment_for_poly(mass, vertices2)
        body = pymunk.Body(mass, inertia)
        shape1, shape2 = pymunk.Poly(body, vertices1), pymunk.Poly(body, vertices2)
        shape1.friction = shape2.friction = 1
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position, body.angle = position, angle
        self.space.add(body, shape1, shape2)
        return body

    def _set_state(self, state: np.ndarray) -> None:
        assert self.space is not None and self.agent is not None and self.block is not None
        self.agent.position = tuple(state[:2])
        if self.legacy:
            self.block.position, self.block.angle = tuple(state[2:4]), float(state[4])
        else:
            self.block.angle, self.block.position = float(state[4]), tuple(state[2:4])
        self.space.step(1.0 / self.sim_hz)

    def _goal_body(self) -> pymunk.Body:
        body = pymunk.Body(1, pymunk.moment_for_box(1, (50, 100)))
        body.position, body.angle = tuple(self.goal_pose[:2]), float(self.goal_pose[2])
        return body

    @staticmethod
    def _draw_circle(canvas: pygame.Surface, body: pymunk.Body | None, radius: int, color: pygame.Color) -> None:
        assert body is not None
        pygame.draw.circle(canvas, color, (round(body.position.x), round(body.position.y)), radius)

    @staticmethod
    def _draw_polygon(canvas: pygame.Surface, body: pymunk.Body, shapes: list[pymunk.Shape], color: pygame.Color) -> None:
        for shape in shapes:
            assert isinstance(shape, pymunk.Poly)
            points = [body.local_to_world(vertex) for vertex in shape.get_vertices()]
            pygame.draw.polygon(canvas, color, [(round(point.x), round(point.y)) for point in points])

    def _handle_collision(self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: object) -> None:
        self.n_contact_points += len(arbiter.contact_point_set.points)
