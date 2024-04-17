import pygame
import numpy as np
from PIL import Image


class Window:
    def __init__(self, img_path: str, img_height: int, img_width: int, K: int) -> None:
        pygame.init()
        self.run: bool = True
        self.WIDTH: int = 1200
        self.HEIGHT: int = 600
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.img_height: int = img_height
        self.img_width: int = img_width
        self.img_array = self.get_resized_array(
            img_path, self.img_height, self.img_width
        )
        self.angle: float = 0
        self.delta_angle: float = 0.01
        self.K: int = K
        self.reroll_means()
        self.clusters: list = [[] for _ in range(self.K)]
        self.recluster()

    def running(self) -> None:
        while self.run:
            self.event()
            self.draw()

    def event(self) -> None:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            self.angle += self.delta_angle
        elif keys[pygame.K_d]:
            self.angle -= self.delta_angle
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reroll_means()
                elif event.key == pygame.K_SPACE:
                    self.move_means()
                    self.recluster()
                elif event.key == pygame.K_s:
                    self.clusters_to_image()

    def draw(self) -> None:
        self.SCREEN.fill((51, 51, 51))
        offset_x: int = self.WIDTH // 4
        offset_y: int = self.HEIGHT * 3 // 4
        clusters_offset_x: int = self.WIDTH * 3 // 4
        clusters_offset_y: int = self.HEIGHT * 3 // 4
        for cluster_idx, cluster in enumerate(self.clusters):
            rotated = self.rotate_points(np.array(cluster), self.angle)
            for projected_point in self.project_points(rotated):
                pygame.draw.circle(
                    self.SCREEN,
                    (
                        self.means[cluster_idx]
                        if self.means is not None
                        else (255, 255, 255)
                    ),
                    (
                        projected_point[0] + clusters_offset_x,
                        -projected_point[1] + clusters_offset_y,
                    ),
                    3,
                )
        for projected_mean, mean in zip(
            self.project_points(self.rotate_points(self.means, self.angle)), self.means
        ):
            pygame.draw.circle(
                self.SCREEN,
                mean,
                (
                    projected_mean[0] + clusters_offset_x,
                    -projected_mean[1] + clusters_offset_y,
                ),
                6,
            )

        for row in self.img_array:
            rotated = self.rotate_points(row, self.angle)
            for point, projected_point in zip(row, self.project_points(rotated)):
                pygame.draw.circle(
                    self.SCREEN,
                    point,
                    (projected_point[0] + offset_x, -projected_point[1] + offset_y),
                    3,
                )
        pygame.display.update()

    @staticmethod
    def get_resized_array(path: str, width: int, height: int) -> np.ndarray:
        img = pygame.image.load(path)
        resized = pygame.transform.scale(img, (width, height))
        return pygame.surfarray.array3d(resized)

    def rotate_points(self, points: np.ndarray, angle: float) -> np.ndarray:
        rotation_matrix = np.array(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )
        return np.matmul(rotation_matrix, points.reshape(len(points), 3, 1))

    def project_points(self, points: np.ndarray) -> np.ndarray:
        return np.matmul(
            np.array([[1, 0, 0], [0, 1, 0]]), points.reshape(len(points), 3, 1)
        ).reshape((len(points), 2))

    @staticmethod
    def lerp(
        x: float | np.ndarray,
        old_min: float,
        old_max: float,
        new_min: float,
        new_max: float,
    ) -> int | np.ndarray:
        return np.round((x - old_min) / (old_max - old_min) * new_max + new_min)

    def reroll_means(self) -> None:
        self.means = np.random.randint(255, size=(self.K, 3))

    def recluster(self) -> None:
        self.clusters = [[] for _ in range(self.K)]
        for row in self.img_array:
            for point in row:
                cluster_idx = np.argmin(np.sum((self.means - point) ** 2, axis=1))
                self.clusters[cluster_idx].append(point)

    def move_means(self) -> None:
        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                self.means[i] = np.sum(cluster, axis=0) / len(cluster)
            else:
                self.means[i] = np.random.randint(255, size=3)

    def clusters_to_image(self):
        new_img_array = np.empty((self.img_height, self.img_width, 3), dtype=np.uint8)
        for y, row in enumerate(self.img_array):
            for x, point in enumerate(row):
                cluster_idx = np.argmin(np.sum((self.means - point) ** 2, axis=1))
                new_img_array[y, x] = self.means[cluster_idx]
        img = Image.fromarray(
            np.concatenate(
                (self.img_array.transpose(1, 0, 2), new_img_array.transpose(1, 0, 2)),
                axis=1,
            ),
            mode="RGB",
        )
        img.show()
