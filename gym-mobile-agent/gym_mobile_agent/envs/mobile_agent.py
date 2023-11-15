import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt


class Object:
    def __init__(self, vertices, type, color=(0, 0, 0)):
        self.vertices = vertices
        self.bounding_box = self._compute_bounding_box()
        self.color = color
        self.type = type

    def get_vertices(self):
        return self.vertices

    def get_color(self):
        return self.color

    def _compute_bounding_box(self):
        """Compute the bounding box of the obstacle"""
        return (
            np.min(self.vertices[:, 0]),
            np.max(self.vertices[:, 0]),
            np.min(self.vertices[:, 1]),
            np.max(self.vertices[:, 1]),
        )

    def _is_inside_bounding_box(self, position, radius):
        """Check if a point is inside the bounding box of the obstacle"""
        return (
            position[0] > self.bounding_box[0] - radius
            and position[0] < self.bounding_box[1] + radius
            and position[1] > self.bounding_box[2] - radius
            and position[1] < self.bounding_box[3] + radius
        )

    def _is_inside_polygon(self, position):
        """Check if a point is inside the polygon"""
        x, y = position
        n = len(self.vertices)
        inside = False
        p1x, p1y = self.vertices[0]
        for i in range(n + 1):
            p2x, p2y = self.vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _intersect_circle(self, position, radius):
        """Check if a circle intersect one of the edges of the polygon"""
        x, y = position
        n = len(self.vertices)
        for i in range(n):
            p1x, p1y = self.vertices[i]
            p2x, p2y = self.vertices[(i + 1) % n]
            x1 = p1x - x
            y1 = p1y - y
            x2 = p2x - x
            y2 = p2y - y
            dx = x2 - x1
            dy = y2 - y1
            dr = np.sqrt(dx**2 + dy**2)
            D = x1 * y2 - x2 * y1
            delta = radius**2 * dr**2 - D**2
            if delta >= 0:
                return True
        return False

    def intersect_segment(self, p1, p2):
        """Check if a segment intersect one of the edges of the polygon"""
        p1x, p1y = p1
        p2x, p2y = p2
        n = len(self.vertices)
        for i in range(n):
            q1x, q1y = self.vertices[i]
            q2x, q2y = self.vertices[(i + 1) % n]
            # solve the system of equations
            # s(p2x - p1x) + t(q1x - q2x) = q1x - p1x
            # s(p2y - p1y) + t(q1y - q2y) = q1y - p1y
            # for s and t
            a1 = p2x - p1x
            b1 = q1x - q2x
            c1 = q1x - p1x
            a2 = p2y - p1y
            b2 = q1y - q2y
            c2 = q1y - p1y
            delta = a1 * b2 - a2 * b1
            if delta != 0:
                s = (c1 * b2 - c2 * b1) / delta
                t = (a1 * c2 - a2 * c1) / delta
                if s >= 0 and s <= 1 and t >= 0 and t <= 1:
                    ix = p1x + s * (p2x - p1x)
                    iy = p1y + s * (p2y - p1y)
                    return True, np.array([ix, iy])

        return False, None

    def is_colliding_circle(self, position, radius):
        """Check if a circle is colliding with an obstacle"""
        # Check if the point is inside the bounding box of the obstacle
        if self._is_inside_bounding_box(position, radius):
            # Check if the point is inside the polygon
            return self._is_inside_polygon(position) or self._intersect_circle(
                position, radius
            )

        else:
            return False

    def is_colliding_object(self, obstacle: "Object"):
        """Check if an obstacle is colliding with another obstacle"""
        n = len(self.vertices)
        for i in range(n):
            p1x, p1y = self.vertices[i]
            p2x, p2y = self.vertices[(i + 1) % n]
            if obstacle.intersect_segment((p1x, p1y), (p2x, p2y))[0]:
                return True
        return False


class Obstacle(Object):
    def __init__(self, vertices, color=(0, 0, 0)):
        super().__init__(vertices, "obstacle", color)


class Wall(Object):
    def __init__(self, vertices, color=(0, 0, 255)):
        super().__init__(vertices, "wall", color)


class Target(Object):
    def __init__(self, position, size, color=(255, 0, 0)):
        vertices = np.array(
            [
                [position[0] - size / 2, position[1] - size / 2],
                [position[0] - size / 2, position[1] + size / 2],
                [position[0] + size / 2, position[1] + size / 2],
                [position[0] + size / 2, position[1] - size / 2],
            ]
        )
        super().__init__(vertices, "target", color)


class MobileAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self, render_mode=None, boundaries=(0, 40, 0, 20), preferred_velocity=1.3
    ) -> None:
        self.render_mode = render_mode
        self.boundaries = boundaries  # (xmin, xmax, ymin, ymax)
        self.window_width = 800
        self.window_height = int(
            self.window_width
            * (boundaries[3] - boundaries[2])
            / (boundaries[1] - boundaries[0])
        )  # keep the same aspect ratio as the environment
        self.max_velocity = 5  # m/s
        self.max_acceleration = 5  # m/s^2
        self.dt = 0.05
        self.tau = 0.5  # relaxation time (s)
        self.preferred_velocity = preferred_velocity  # m/s
        self.steering_force_magnitude = 10  # N

        self.radius = 0.3  # m

        self.fov_angle = 100  # degrees
        self.fov_angle_rad = np.deg2rad(self.fov_angle)
        self.fov_bin_number = 400  # number of bins in the field of view
        self.max_view_distance = 10  # m

        # Define the agent's location, velocity and acceleration
        self._agent_location = None
        self._agent_velocity = None
        self._agent_acceleration = None

        self.target_size = 0.5  # m

        # Define the walls of the environment
        self.wall_thickness = 0.5
        self.walls = self._compute_walls()

        # Define some obstacles
        self.n_obstacles = 5
        self.obstacles = []

        self.targets = []

        # Observations are the position of the pedestrian, the position of the
        # goal, and the field of view of the pedestrian
        # field of view is a 1D array (Nfov, 1) where Nfov covers an arc of the
        # pedestrian's field of view (from left,  -100 degrees to right, 100 degrees).
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, self.fov_bin_number, 4), dtype=np.float32
        )
        #     {
        #         "agent_position": spaces.Box(
        #             low=np.array([boundaries[0], boundaries[2]]),
        #             high=np.array([boundaries[1], boundaries[3]]),
        #             dtype=np.float32,
        #         ),
        #         "agent_velocity": spaces.Box(
        #             low=np.array([-self.max_velocity, -self.max_velocity]),
        #             high=np.array([self.max_velocity, self.max_velocity]),
        #             dtype=np.float32,
        #         ),
        #         "agent_acceleration": spaces.Box(
        #             low=np.array([-self.max_acceleration, -self.max_acceleration]),
        #             high=np.array([self.max_acceleration, self.max_acceleration]),
        #         ),
        #         "agent_fov": spaces.Box(
        #             low=0, high=255, shape=(1, self.fov_bin_number, 4), dtype=np.float32
        #         ),
        #         "goal_position": spaces.Box(
        #             low=np.array([boundaries[0], boundaries[2]]),
        #             high=np.array([boundaries[1], boundaries[3]]),
        #             dtype=np.float32,
        #         ),
        #     }
        # )

        # actions are the new steering force applied to the pedestrian
        self.action_space = spaces.Discrete(9)
        self._action_to_force = {
            0: np.array([0, 0]),  # no steering force
            1: np.array([-1, -1]) / np.sqrt(2),  # left-down
            2: np.array([0, -1]),  # down
            3: np.array([1, -1]) / np.sqrt(2),  # right-down
            4: np.array([-1, 0]),  # left
            5: np.array([0, 0]),  # no steering force
            6: np.array([1, 0]),  # right
            7: np.array([-1, 1]) / np.sqrt(2),  # left-up
            8: np.array([0, 1]),  # up
            9: np.array([1, 1]) / np.sqrt(2),  # right-up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _compute_random_obstacles(self, n_obstacles):
        x_min, x_max, y_min, y_max = self.boundaries
        x_min += self.wall_thickness
        x_max -= self.wall_thickness
        y_min += self.wall_thickness
        y_max -= self.wall_thickness
        obstacles = []
        while len(obstacles) < n_obstacles:
            x = self.np_random.uniform(x_min, x_max)
            y = self.np_random.uniform(y_min, y_max)
            size = self.np_random.uniform(0.5, 2)
            vertices = np.array(
                [
                    [x - size / 2, y - size / 2],
                    [x - size / 2, y + size / 2],
                    [x + size / 2, y + size / 2],
                    [x + size / 2, y - size / 2],
                ]
            )
            obstacle = Obstacle(vertices)
            # check if the obstacle is not colliding with another obstacle
            collision = False
            for object in obstacles + self.walls:
                if obstacle.is_colliding_object(object):
                    collision = True
                    break
            if not collision:
                obstacles.append(obstacle)
        return obstacles

    def _compute_walls(self):
        return [
            Wall(
                np.array(
                    [
                        [self.boundaries[0], self.boundaries[2]],
                        [self.boundaries[0], self.boundaries[2] + self.wall_thickness],
                        [self.boundaries[1], self.boundaries[2] + self.wall_thickness],
                        [self.boundaries[1], self.boundaries[2]],
                    ]
                )
            ),
            Wall(
                np.array(
                    [
                        [self.boundaries[0], self.boundaries[3]],
                        [self.boundaries[0], self.boundaries[3] - self.wall_thickness],
                        [self.boundaries[1], self.boundaries[3] - self.wall_thickness],
                        [self.boundaries[1], self.boundaries[3]],
                    ]
                )
            ),
            Wall(
                np.array(
                    [
                        [self.boundaries[0], self.boundaries[2]],
                        [self.boundaries[0] + self.wall_thickness, self.boundaries[2]],
                        [self.boundaries[0] + self.wall_thickness, self.boundaries[3]],
                        [self.boundaries[0], self.boundaries[3]],
                    ]
                )
            ),
            Wall(
                np.array(
                    [
                        [self.boundaries[1], self.boundaries[2]],
                        [self.boundaries[1] - self.wall_thickness, self.boundaries[2]],
                        [self.boundaries[1] - self.wall_thickness, self.boundaries[3]],
                        [self.boundaries[1], self.boundaries[3]],
                    ]
                )
            ),
        ]

    @property
    def objects(self):
        return self.walls + self.obstacles + self.targets

    def _get_obs(self):
        return self._agent_fov
        # return {
        #     "agent_position": self._agent_location,
        #     "agent_velocity": self._agent_velocity,
        #     "agent_acceleration": self._agent_acceleration,
        #     "agent_fov": self._agent_fov,
        #     "goal_position": self._target_location,
        # }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location)
        }

    def _get_random_position(self):
        x_min, x_max, y_min, y_max = self.boundaries
        return np.array(
            [
                self.np_random.uniform(x_min, x_max),
                self.np_random.uniform(y_min, y_max),
            ],
            dtype=np.float32,
        )

    def _get_random_direction(self):
        random_dir = np.array(
            [
                self.np_random.uniform(-1, 1),
                self.np_random.uniform(-1, 1),
            ],
            dtype=np.float32,
        )
        return random_dir / np.linalg.norm(random_dir)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        while True:
            random_agent_location = self._get_random_position()
            collision = False
            for object in self.objects:
                if object.is_colliding_circle(random_agent_location, self.radius):
                    collision = True
                    break
            if not collision:
                break
        self._agent_location = random_agent_location

        # Set the agent's velocity to the preferred velocity in a random direction
        self._agent_velocity = self.preferred_velocity * self._get_random_direction()
        # self._agent_velocity = np.array([1, 0], dtype=np.float32)

        # Set the agent's acceleration to zero
        self._agent_acceleration = np.zeros(2, dtype=np.float32)

        # Set the agent's field of view
        self._agent_fov = self._get_fov()

        # Choose the obstacles' locations at random
        self.obstacles = self._compute_random_obstacles(self.n_obstacles)

        # Choose the target's location uniformly at random
        while True:
            random_target_location = self._get_random_position()
            target = Target(random_target_location, self.target_size)
            collision = False
            for object in self.objects:
                if target.is_colliding_object(object):
                    collision = True
                    break
            if not collision:
                break
        self._target_location = random_target_location
        self.targets = [target]

        observation = self._get_obs()
        info = self._get_info()

        # print(observation)

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def _get_fov(self):
        # the content of the field of view is a 1D array (Nfov, 1, 4)
        # it is computed by tracing a line from the agent's position at various
        # angles and checking if the line intersects with an object
        # the value of the array is (r, g, b, a) where r, g, b is the color of
        # the object and a is the distance to the object

        fov = np.full((1, self.fov_bin_number, 4), 255, dtype=np.float32)
        fov[:, :, 3] = self.max_view_distance
        direction = self._agent_velocity / np.linalg.norm(self._agent_velocity)
        for i in range(self.fov_bin_number):
            angle = (
                self.fov_angle_rad / 2 - i * self.fov_angle_rad / self.fov_bin_number
            )
            # angle on both sides of the direction
            line_start = self._agent_location
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            line_end = (
                self._agent_location
                + (rotation_matrix @ direction) * self.max_view_distance
            )
            for object in self.objects:
                intersect, intersection = object.intersect_segment(line_start, line_end)
                if intersect:
                    distance = np.linalg.norm(intersection - line_start)
                    if distance < fov[0, i, 3]:  # only see the closest obstacle
                        fov[0, i, :3] = object.get_color()
                        fov[0, i, 3] = distance
        # normalize the alpha channel
        fov[:, :, 3] = fov[:, :, 3] / self.max_view_distance * 255
        return fov

    def step(self, action):
        self._agent_acceleration = np.zeros(2, dtype=np.float32)

        # Find the steering force based on the action
        steering_force = self._action_to_force[action]

        # Normalize the steering force
        force_mag = np.linalg.norm(steering_force)
        if force_mag > 0:
            steering_force = steering_force / force_mag

        steering_force *= self.steering_force_magnitude

        # Update the position, velocity and acceleration of the agent
        self._agent_acceleration += steering_force

        acceleration_magnitude = np.linalg.norm(self._agent_acceleration)
        if acceleration_magnitude > self.max_acceleration:
            self._agent_acceleration = (
                self._agent_acceleration
                / acceleration_magnitude
                * self.max_acceleration
            )

        # Update the velocity of the agent
        self._agent_velocity += self._agent_acceleration * self.dt

        # Limit the velocity of the agent
        velocity_magnitude = np.linalg.norm(self._agent_velocity)
        if velocity_magnitude > self.max_velocity:
            self._agent_velocity = (
                self._agent_velocity / velocity_magnitude * self.max_velocity
            )

        # Update the position of the agent
        self._agent_location += self._agent_velocity * self.dt

        # Update the field of view of the agent
        self._agent_fov = self._get_fov()

        # fig, ax = plt.subplots(figsize=(12, 8))
        # image_fov = np.tile(self._agent_fov, [100, 1])
        # print(image_fov.shape)
        # ax.imshow(image_fov, cmap="Greys", vmin=0, vmax=1)
        # ax.set_xlabel("Angle (°)")
        # ax.set_ylabel("Distance (m)")
        # plt.show()

        reward = 0

        # Check if the agent has reached the target or if it has collided with
        # an obstacle
        done = False
        for object in self.objects:
            if object.is_colliding_circle(self._agent_location, self.radius):
                if object.type == "target":
                    reward += 1000
                else:
                    reward -= 100
                done = True
                break

        # Compute the reward
        # Penalty for having a velocity different from the preferred velocity
        reward -= np.abs(np.linalg.norm(self._agent_velocity) - self.preferred_velocity)
        # Penalty for strong accelerations
        # reward -= np.linalg.norm(acceleration_vector)
        # Penalty for being far from the target
        distance = np.linalg.norm(self._agent_location - self._target_location)
        reward -= distance

        reward = reward.astype(np.float32)

        if self.render_mode == "human":
            self._render_frame()

        # print(self._get_obs())
        # exit()

        return self._get_obs(), reward, done, False, self._get_info()

    def _to_pixels_position(self, position):
        return np.array(
            [
                position[0]
                / (self.boundaries[1] - self.boundaries[0])
                * self.window_width,
                (1 - position[1] / (self.boundaries[3] - self.boundaries[2]))
                * self.window_height,
            ],
            dtype=np.int32,
        )

    def _to_pixels_distance(self, distance):
        return distance / (self.boundaries[1] - self.boundaries[0]) * self.window_width

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._to_pixels_position(self._agent_location),
            self._to_pixels_distance(self.radius),
        )
        # Draw agent's velocity
        pygame.draw.line(
            canvas,
            (0, 0, 255),
            self._to_pixels_position(self._agent_location),
            self._to_pixels_position(self._agent_location + self._agent_velocity),
            2,
        )

        # Draw the agent's field of view (arc of circle)
        boundary_rect = pygame.Rect(  # [left, top], [width, height]
            self._to_pixels_position(
                self._agent_location
                + np.array([-self.max_view_distance, self.max_view_distance])
            ),
            np.array(
                [
                    self._to_pixels_distance(2 * self.max_view_distance),
                    self._to_pixels_distance(2 * self.max_view_distance),
                ]
            ),
        )
        direction_angle = np.arctan2(self._agent_velocity[1], self._agent_velocity[0])
        min_arc_angle = direction_angle - self.fov_angle_rad / 2
        max_arc_angle = direction_angle + self.fov_angle_rad / 2
        pygame.draw.arc(
            canvas,
            (0, 0, 255),
            boundary_rect,
            min_arc_angle,
            max_arc_angle,
            2,
        )
        # line from agent to each end of the arc
        pygame.draw.line(
            canvas,
            (0, 0, 255),
            self._to_pixels_position(self._agent_location),
            self._to_pixels_position(
                self._agent_location
                + np.array(
                    [
                        self.max_view_distance * np.cos(min_arc_angle),
                        self.max_view_distance * np.sin(min_arc_angle),
                    ]
                )
            ),
            2,
        )
        pygame.draw.line(
            canvas,
            (0, 0, 255),
            self._to_pixels_position(self._agent_location),
            self._to_pixels_position(
                self._agent_location
                + np.array(
                    [
                        self.max_view_distance * np.cos(max_arc_angle),
                        self.max_view_distance * np.sin(max_arc_angle),
                    ]
                )
            ),
            2,
        )

        # Draw the obstacles
        for object in self.objects:
            pygame.draw.polygon(
                canvas,
                object.get_color(),
                [self._to_pixels_position(vertex) for vertex in object.get_vertices()],
            )

        # Draw the content of the field of view (1D array)
        # at the bottom right of the window
        n_vertical_pixels = 30
        border_width = 10
        tiled_fov = np.tile(
            self._agent_fov[:, :, :3], (n_vertical_pixels, 1, 1)
        ).astype(np.uint8)
        image_fov = np.zeros(
            (
                n_vertical_pixels + 2 * border_width,
                self.fov_bin_number + 2 * border_width,
                3,
            ),
            dtype=np.uint8,
        )
        image_fov[border_width:-border_width, border_width:-border_width, :] = tiled_fov
        fov_surface = pygame.surfarray.make_surface(
            np.transpose(image_fov, axes=(1, 0, 2))
        )
        # resize the image to make it smaller
        width_surface = int(self.window_width / 7)
        height_surface = int(width_surface * image_fov.shape[0] / image_fov.shape[1])
        scale_factor = width_surface / image_fov.shape[1]
        fov_surface = pygame.transform.scale_by(fov_surface, scale_factor)
        canvas.blit(
            fov_surface,
            (self.window_width - width_surface, self.window_height - height_surface),
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
