"""Лабораторная работа: Boids, вариант 1 — стая птиц в 2D.

Запуск анимации:              python boids_birds.py
Эксперименты и изображение:   python boids_birds.py --experiments
Проверка без окна:           python boids_birds.py --check

Зависимости: numpy, matplotlib.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


# 1. ПАРАМЕТРЫ СРЕДЫ И АГЕНТОВ (можно изменять)
WIDTH = 100
HEIGHT = 100
BIRD_COUNT = 30              # Не менее 20 по заданию
STEPS = 600                  # Не менее 500 по заданию
NEIGHBOR_RADIUS = 18         # Радиус поиска соседей
SEPARATION_RADIUS = 6        # Дистанция, на которой птицы разлетаются
MAX_SPEED = 1.6              # Максимальная скорость
MIN_SPEED = 0.8              # Чтобы стая не останавливалась
MAX_ACCELERATION = 0.16      # Максимальное изменение скорости за шаг

# Веса трёх правил Boids
COHESION_WEIGHT = 0.01       # Центрирование: двигаться к соседям
ALIGNMENT_WEIGHT = 0.05      # Выравнивание: лететь в одном направлении
SEPARATION_WEIGHT = 0.10     # Разделение: не сближаться слишком сильно
RANDOM_SEED = 42             # Для повторяемых экспериментов


class Flock:
    """Модель стаи: положение, скорость и обновление птиц."""

    def __init__(self, radius=NEIGHBOR_RADIUS,
                 cohesion=COHESION_WEIGHT,
                 alignment=ALIGNMENT_WEIGHT,
                 separation=SEPARATION_WEIGHT,
                 seed=RANDOM_SEED):
        self.radius = radius
        self.cohesion = cohesion
        self.alignment = alignment
        self.separation = separation
        rng = np.random.default_rng(seed)

        # Случайные начальные положения и направления 30 птиц.
        self.positions = rng.uniform([0, 0], [WIDTH, HEIGHT],
                                     size=(BIRD_COUNT, 2))
        angles = rng.uniform(0, 2 * np.pi, BIRD_COUNT)
        self.velocities = np.column_stack((np.cos(angles), np.sin(angles)))

    @staticmethod
    def limit(vector, maximum):
        """Ограничивает длину вектора (скорость или ускорение)."""
        length = np.linalg.norm(vector)
        if length > maximum:
            return vector / length * maximum
        return vector

    def offsets_from(self, i):
        """Смещения других птиц относительно i с учётом циклических границ."""
        size = np.array([WIDTH, HEIGHT])
        return (self.positions - self.positions[i] + size / 2) % size - size / 2

    def step(self):
        """Один шаг модели: применить три правила и обновить координаты."""
        new_velocities = self.velocities.copy()

        for i in range(BIRD_COUNT):
            offsets = self.offsets_from(i)
            distances = np.linalg.norm(offsets, axis=1)

            # Соседи — другие птицы в заданном радиусе.
            neighbors = (distances > 0) & (distances < self.radius)
            acceleration = np.zeros(2)

            if np.any(neighbors):
                # 1. Центрирование — движение к центру ближайших соседей.
                cohesion_force = offsets[neighbors].mean(axis=0)

                # 2. Выравнивание — приближение скорости к средней скорости соседей.
                alignment_force = (self.velocities[neighbors].mean(axis=0)
                                   - self.velocities[i])

                # 3. Разделение — отталкивание от слишком близких птиц.
                close = neighbors & (distances < SEPARATION_RADIUS)
                separation_force = np.zeros(2)
                if np.any(close):
                    separation_force = -np.sum(
                        offsets[close] / (distances[close, None] ** 2 + 1e-6),
                        axis=0,
                    )

                # Суммируем три вектора с разными весами.
                acceleration = (self.cohesion * cohesion_force
                                + self.alignment * alignment_force
                                + self.separation * separation_force)

            acceleration = self.limit(acceleration, MAX_ACCELERATION)
            velocity = self.limit(self.velocities[i] + acceleration, MAX_SPEED)
            # Сохраняем минимальную скорость, чтобы птицы продолжали лететь.
            speed = np.linalg.norm(velocity)
            if 0 < speed < MIN_SPEED:
                velocity = velocity / speed * MIN_SPEED
            new_velocities[i] = velocity

        # Все птицы меняют положение одновременно.
        self.velocities = new_velocities
        self.positions = (self.positions + self.velocities) % [WIDTH, HEIGHT]

    def statistics(self):
        """Численные показатели, позволяющие сравнить эксперименты."""
        speed = np.linalg.norm(self.velocities, axis=1)
        unit_velocity = self.velocities / np.maximum(speed[:, None], 1e-9)
        alignment_score = np.linalg.norm(unit_velocity.mean(axis=0))

        neighbors_total = 0
        very_close_pairs = 0
        for i in range(BIRD_COUNT):
            distances = np.linalg.norm(self.offsets_from(i), axis=1)
            neighbors_total += np.count_nonzero(
                (distances > 0) & (distances < self.radius)
            )
            # Каждую пару считаем только один раз (j > i).
            very_close_pairs += np.count_nonzero(
                (distances[i + 1:] < 2.0)
            )

        return (alignment_score,
                neighbors_total / BIRD_COUNT,
                very_close_pairs)


def save_snapshot(flock, filepath):
    """Сохраняет один кадр для раздела «Результаты визуализации»."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(flock.positions[:, 0], flock.positions[:, 1],
              flock.velocities[:, 0], flock.velocities[:, 1],
              angles='xy', scale_units='xy', scale=0.23, width=0.005, color='royalblue')
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT),
           xlabel='X', ylabel='Y', title=f'Boids: стая после {STEPS} шагов')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    fig.savefig(filepath, dpi=160, bbox_inches='tight')
    plt.close(fig)


def run_experiments():
    """Сравнивает радиусы и веса. Сохраняет таблицу и итоговый кадр."""
    output = Path(__file__).resolve().parent
    configs = [
        ('Базовый', 18, 0.01, 0.05, 0.10),
        ('Малый радиус', 10, 0.01, 0.05, 0.10),
        ('Большой радиус', 30, 0.01, 0.05, 0.10),
        ('Слабое разделение', 18, 0.01, 0.05, 0.03),
        ('Сильное разделение', 18, 0.01, 0.05, 0.25),
    ]
    rows = []

    for name, radius, cohesion, alignment, separation in configs:
        flock = Flock(radius, cohesion, alignment, separation)
        scores = []
        neighbor_counts = []
        close_counts = []
        for step in range(STEPS):
            flock.step()
            # Берём среднее за последние 100 шагов, а не случайный один кадр.
            if step >= STEPS - 100:
                score, neighbors, close = flock.statistics()
                scores.append(score)
                neighbor_counts.append(neighbors)
                close_counts.append(close)

        rows.append({
            'Эксперимент': name,
            'Шагов': STEPS,
            'Птиц': BIRD_COUNT,
            'Радиус': radius,
            'Вес центрирования': cohesion,
            'Вес выравнивания': alignment,
            'Вес разделения': separation,
            'Согласованность направлений (0-1)': round(float(np.mean(scores)), 3),
            'Среднее число соседей': round(float(np.mean(neighbor_counts)), 2),
            'Пар ближе 2 ед.': round(float(np.mean(close_counts)), 2),
        })

        if name == 'Базовый':
            save_snapshot(flock, output / 'boids_step600.png')

    file_path = output / 'boids_experiments.csv'
    with file_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f'Эксперименты завершены. Таблица: {file_path.name}')
    print('Кадр базового эксперимента: boids_step600.png')
    for row in rows:
        print(f"{row['Эксперимент']:<22} "
              f"согласованность={row['Согласованность направлений (0-1)']:.3f}, "
              f"соседей={row['Среднее число соседей']:.2f}, "
              f"близких пар={row['Пар ближе 2 ед.']:.2f}")


def animate():
    """Анимация: синий треугольник/стрелка показывает птицу и её направление."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    flock = Flock()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set(xlim=(0, WIDTH), ylim=(0, HEIGHT),
           xlabel='X', ylabel='Y')
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)

    arrows = ax.quiver(flock.positions[:, 0], flock.positions[:, 1],
                       flock.velocities[:, 0], flock.velocities[:, 1],
                       angles='xy', scale_units='xy', scale=0.23, width=0.005,
                       color='royalblue')
    title = ax.set_title('Стая птиц (Boids) — шаг 0')
    paused = [False]

    def update(frame):
        flock.step()
        arrows.set_offsets(flock.positions)
        arrows.set_UVC(flock.velocities[:, 0], flock.velocities[:, 1])
        title.set_text(f'Стая птиц (Boids) — шаг {frame} из {STEPS}')
        return arrows, title

    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
            if paused[0]:
                animation.event_source.stop()
            else:
                animation.event_source.start()

    animation = FuncAnimation(fig, update, frames=range(1, STEPS + 1),
                              interval=35, blit=False, repeat=False)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Boids: вариант 1, стая птиц')
    parser.add_argument('--experiments', action='store_true',
                        help='создать CSV экспериментов и PNG-кадр')
    parser.add_argument('--check', action='store_true',
                        help='проверить 600 шагов без графического окна')
    args = parser.parse_args()

    if args.experiments:
        run_experiments()
    elif args.check:
        flock = Flock()
        for _ in range(STEPS):
            flock.step()
        assert np.all(np.isfinite(flock.positions))
        assert np.max(np.linalg.norm(flock.velocities, axis=1)) <= MAX_SPEED + 1e-8
        print(f'Проверка пройдена: {BIRD_COUNT} птиц, {STEPS} шагов, скорость ограничена.')
    else:
        animate()


if __name__ == '__main__':
    main()
