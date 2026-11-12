"""ЛР №5. Генетический алгоритм для 4-битных хромосом.

Запуск всех экспериментов и создание README.md:
    python genetic_algorithm.py

Собственные параметры (один запуск):
    python genetic_algorithm.py --single --variant 1 --n 8 --pc 0.8 --pm 0.05 --g 40
"""

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Сохранение графиков без открытия окна
import matplotlib.pyplot as plt

L = 4  # Длина хромосомы по условию всегда равна четырём
MAX_VALUE = 2 ** L - 1

# Все варианты поддерживаются, но по умолчанию используется вариант №1.
FUNCTIONS = {
    1: lambda x: x * math.sin(6 * math.pi * x) + 0.5 * math.cos(4 * math.pi * x) + 1,
    2: lambda x: math.exp(-x) * math.sin(5 * math.pi * x),
    3: lambda x: math.sin(4 * math.pi * x) ** 2 + 0.3 * x,
    4: lambda x: x * math.cos(3 * math.pi * x) + 0.2,
    5: lambda x: math.sqrt(x) * math.sin(8 * math.pi * x) + 1,
    6: lambda x: math.cos(5 * math.pi * x) + x ** 2,
    7: lambda x: math.exp(-(x - 0.5) ** 2),
    8: lambda x: 1 - abs(2 * x - 1),
    9: lambda x: x * (1 - x) * math.sin(10 * math.pi * x) + 1,
    10: lambda x: math.sin(2 * math.pi * x) + 0.5 * math.sin(5 * math.pi * x) + 1,
    11: lambda x: math.exp(-10 * x) + 0.5 * x,
    12: lambda x: x * math.sin(10 * x),
    13: lambda x: 0.5 * math.cos(6 * math.pi * x) + x ** 3,
    14: lambda x: abs(x - 0.3) * math.sin(4 * math.pi * x) + 1,
    15: lambda x: math.exp(-x ** 2) * math.cos(3 * math.pi * x),
    16: lambda x: math.sin(3 * math.pi * x) + 0.2 * x ** 2,
    17: lambda x: x * math.log(x + 1) + 1,
    18: lambda x: math.sin(6 * math.pi * x ** 2) + 1,
    19: lambda x: (1 - x) * math.sin(5 * math.pi * x) + 1,
    20: lambda x: math.exp(-5 * abs(x - 0.5)),
    21: lambda x: x ** 2 * math.cos(5 * math.pi * x) + 1,
    22: lambda x: math.sin(7 * math.pi * x) + 0.5 * math.cos(3 * math.pi * x) + 1,
    23: lambda x: x ** 3 - 2 * x ** 2 + 0.5 * math.sin(4 * math.pi * x) + 1,
    24: lambda x: math.exp(-2 * x) * math.sin(4 * math.pi * x) + 1,
    25: lambda x: 0.7 * math.cos(8 * math.pi * x) + 0.3 * x,
}
FORMULAS = {
    1: r"f(x) = x\sin(6\pi x) + 0.5\cos(4\pi x) + 1",
    2: r"f(x) = e^{-x}\sin(5\pi x)",
    3: r"f(x) = \sin^2(4\pi x) + 0.3x",
    4: r"f(x) = x\cos(3\pi x)+0.2",
    5: r"f(x) = \sqrt{x}\sin(8\pi x)+1",
    6: r"f(x) = \cos(5\pi x)+x^2",
    7: r"f(x) = e^{-(x-0.5)^2}",
    8: r"f(x) = 1-|2x-1|",
    9: r"f(x) = x(1-x)\sin(10\pi x)+1",
    10: r"f(x) = \sin(2\pi x)+0.5\sin(5\pi x)+1",
    11: r"f(x) = e^{-10x}+0.5x",
    12: r"f(x) = x\sin(10x)",
    13: r"f(x) = 0.5\cos(6\pi x)+x^3",
    14: r"f(x) = |x-0.3|\sin(4\pi x)+1",
    15: r"f(x) = e^{-x^2}\cos(3\pi x)",
    16: r"f(x) = \sin(3\pi x)+0.2x^2",
    17: r"f(x) = x\ln(x+1)+1",
    18: r"f(x) = \sin(6\pi x^2)+1",
    19: r"f(x) = (1-x)\sin(5\pi x)+1",
    20: r"f(x) = e^{-5|x-0.5|}",
    21: r"f(x) = x^2\cos(5\pi x)+1",
    22: r"f(x) = \sin(7\pi x)+0.5\cos(3\pi x)+1",
    23: r"f(x) = x^3-2x^2+0.5\sin(4\pi x)+1",
    24: r"f(x) = e^{-2x}\sin(4\pi x)+1",
    25: r"f(x) = 0.7\cos(8\pi x)+0.3x",
}


@dataclass(frozen=True)
class Params:
    n: int = 4
    pc: float = 0.8
    pm: float = 0.05
    g: int = 30
    seed: int = 154
    variant: int = 1

    def validate(self):
        if not 2 <= self.n <= 20:
            raise ValueError("Размер популяции N должен быть от 2 до 20")
        if not 0 <= self.pc <= 1 or not 0 <= self.pm <= 1:
            raise ValueError("Вероятности pc и pm должны быть от 0 до 1")
        if self.g < 30:
            raise ValueError("Количество поколений G должно быть не менее 30")
        if self.variant not in FUNCTIONS:
            raise ValueError("Вариант функции должен быть от 1 до 25")


def bits(number):
    """Превращает десятичное число в двоичную строку из 4 бит."""
    return format(number, f"0{L}b")


def decode(number):
    """0000 -> 0, 1111 -> 1; остальные равномерно между ними."""
    return number / MAX_VALUE


def fitness(number, variant):
    return FUNCTIONS[variant](decode(number))


def tournament(population, values, rng):
    """Случайно выбираем 2 кандидатов, побеждает более приспособленный."""
    a, b = rng.randrange(len(population)), rng.randrange(len(population))
    return a if values[a] >= values[b] else b


def crossover(a, b, pc, rng):
    """Одноточечное скрещивание двух четырехбитных хромосом."""
    if rng.random() >= pc:
        return a, b, None
    point = rng.randint(1, L - 1)  # Точка после 1-го, 2-го или 3-го бита
    mask_right = (1 << (L - point)) - 1
    child_1 = (a & ~mask_right) | (b & mask_right)
    child_2 = (b & ~mask_right) | (a & mask_right)
    return child_1, child_2, point


def mutate(number, pm, rng):
    """Каждый из 4 битов меняется с независимой вероятностью pm."""
    changes = []
    for position in range(1, L + 1):  # нумерация битов слева направо
        if rng.random() < pm:
            mask = 1 << (L - position)
            old_bit = 1 if number & mask else 0
            number ^= mask
            changes.append((position, old_bit, 1 - old_bit))
    return number, changes


def exact_discrete_optimum(variant):
    """Проверка всех 16 кодов, чтобы знать точный максимум НА СЕТКЕ."""
    best_code = max(range(MAX_VALUE + 1), key=lambda b: fitness(b, variant))
    return best_code, fitness(best_code, variant)


def run_algorithm(params, detailed=True):
    params.validate()
    rng = random.Random(params.seed)
    population = [rng.randrange(MAX_VALUE + 1) for _ in range(params.n)]
    optimum_code, optimum_f = exact_discrete_optimum(params.variant)
    rows, summary, log_lines = [], [], []
    best_code = None
    best_fitness = float("-inf")
    first_hit = None

    for generation in range(1, params.g + 1):
        values = [fitness(ch, params.variant) for ch in population]
        generation_best_index = max(range(params.n), key=lambda i: values[i])
        generation_best = population[generation_best_index]
        gen_max = max(values)
        gen_min = min(values)
        gen_mean = statistics.mean(values)

        if gen_max > best_fitness:
            best_fitness, best_code = gen_max, generation_best
        if first_hit is None and any(math.isclose(v, optimum_f, rel_tol=1e-12, abs_tol=1e-12) for v in values):
            first_hit = generation

        summary.append({
            "generation": generation,
            "min_fitness": gen_min,
            "mean_fitness": gen_mean,
            "max_fitness": gen_max,
            "best_ever_fitness": best_fitness,
            "unique_chromosomes": len(set(population)),
        })
        if detailed:
            log_lines.extend([f"Поколение: {generation}", "Популяция:"])
            for index, (chromosome, value) in enumerate(zip(population, values), 1):
                x = decode(chromosome)
                rows.append({
                    "generation": generation,
                    "individual": index,
                    "chromosome": bits(chromosome),
                    "decimal": chromosome,
                    "x": x,
                    "fitness": value,
                })
                log_lines.append(
                    f"  {index}. Хромосома: {bits(chromosome)} | "
                    f"Десятичное значение: {chromosome:2d} | x: {x:.6f} | fitness: {value:.6f}"
                )
            log_lines.extend([
                "Статистика поколения:",
                f"  Минимальный fitness: {gen_min:.6f}",
                f"  Средний fitness:   {gen_mean:.6f}",
                f"  Максимальный fitness: {gen_max:.6f}",
                f"  Уникальных хромосом: {len(set(population))} из {params.n}",
                f"  Лучшая особь поколения: №{generation_best_index + 1} ({bits(generation_best)})",
            ])

        if generation == params.g:
            if detailed:
                log_lines.extend([
                    "Операторы: это последнее поколение; новая популяция уже не создаётся.",
                    "------------------------------------------------------------", "",
                ])
            break

        # Элитизм: переносим лучшую хромосому без изменений в следующее поколение.
        next_population = [generation_best]
        if detailed:
            log_lines.extend([
                "Формирование следующего поколения:",
                f"  Элитизм: особь №{generation_best_index + 1} ({bits(generation_best)}) перенесена без изменений.",
            ])
        pair_number = 0
        while len(next_population) < params.n:
            pair_number += 1
            first_index = tournament(population, values, rng)
            second_index = tournament(population, values, rng)
            parent_1, parent_2 = population[first_index], population[second_index]
            child_1, child_2, point = crossover(parent_1, parent_2, params.pc, rng)
            if detailed:
                log_lines.extend([
                    f"  Пара родителей №{pair_number} (турнирная селекция):",
                    f"    Родитель 1: особь №{first_index + 1}, {bits(parent_1)}, fitness={values[first_index]:.6f}",
                    f"    Родитель 2: особь №{second_index + 1}, {bits(parent_2)}, fitness={values[second_index]:.6f}",
                ])
                if point is not None:
                    log_lines.append(
                        f"    Кроссовер: ДА; точка после бита {point}; "
                        f"потомки до мутации: {bits(child_1)} и {bits(child_2)}"
                    )
                else:
                    log_lines.append(
                        f"    Кроссовер: НЕТ; потомки-копии: {bits(child_1)} и {bits(child_2)}; точка: —"
                    )

            # Если N нечётно, неиспользованный последний потомок НЕ мутирует.
            for child in (child_1, child_2):
                if len(next_population) >= params.n:
                    break
                position_in_new_generation = len(next_population) + 1
                new_child, changes = mutate(child, params.pm, rng)
                next_population.append(new_child)
                if detailed:
                    if not changes:
                        log_lines.append(
                            f"    Мутация в новой особи №{position_in_new_generation}: НЕТ; итог: {bits(new_child)}"
                        )
                    else:
                        for bit_position, before, after in changes:
                            log_lines.append(
                                f"    Мутация в новой особи №{position_in_new_generation}, "
                                f"бит {bit_position} изменён с {before} на {after}"
                            )
                        log_lines.append(
                            f"    Итог после мутации, новая особь №{position_in_new_generation}: {bits(new_child)}"
                        )
        population = next_population
        if detailed:
            log_lines.extend([
                "Следующее поколение: " + ", ".join(bits(b) for b in population),
                "------------------------------------------------------------", "",
            ])

    return {
        "params": params,
        "rows": rows,
        "stats": summary,
        "log": "\n".join(log_lines) + "\n" if detailed else "",
        "best_code": best_code,
        "best_fitness": best_fitness,
        "first_hit": first_hit,
        "optimum_code": optimum_code,
        "optimum_fitness": optimum_f,
    }


def write_csv(path, rows, fieldnames, decimals=8):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{row[key]:.{decimals}f}" if isinstance(row[key], float) else row[key]
                for key in fieldnames
            })


def export_experiment(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "full_log.txt").write_text(result["log"], encoding="utf-8")
    write_csv(folder / "evolution.csv", result["rows"],
              ["generation", "individual", "chromosome", "decimal", "x", "fitness"])
    write_csv(folder / "statistics.csv", result["stats"],
              ["generation", "min_fitness", "mean_fitness", "max_fitness", "best_ever_fitness", "unique_chromosomes"])

    stats = result["stats"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([r["generation"] for r in stats], [r["max_fitness"] for r in stats],
            marker="o", markersize=3, linewidth=1.8, label="Максимальный fitness поколения")
    ax.plot([r["generation"] for r in stats], [r["mean_fitness"] for r in stats],
            linewidth=1.3, alpha=0.65, label="Средний fitness поколения")
    ax.axhline(result["optimum_fitness"], linestyle="--", linewidth=1.3,
               label="Точный максимум на сетке из 16 значений")
    ax.set(title="Динамика fitness по поколениям", xlabel="Поколение", ylabel="Fitness")
    ax.set_xlim(1, result["params"].g)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(folder / "fitness.png", dpi=160)
    plt.close(fig)


def markdown_table(headers, rows):
    clean = lambda value: str(value).replace("|", "\\|")
    return ("| " + " | ".join(map(clean, headers)) + " |\n" +
            "| " + " | ".join("---" for _ in headers) + " |\n" +
            "".join("| " + " | ".join(map(clean, row)) + " |\n" for row in rows))


def repeat_experiments(experiments, repetitions, variant):
    """Повторяем для статистики: разными seed, одинаковыми seed для каждого набора параметров."""
    table = []
    for experiment in experiments:
        best_values, first_hits, unique_final = [], [], []
        for j in range(repetitions):
            p = Params(n=experiment.n, pc=experiment.pc, pm=experiment.pm, g=experiment.g,
                       seed=300000 + j, variant=variant)
            result = run_algorithm(p, detailed=False)
            best_values.append(result["best_fitness"])
            if result["first_hit"] is not None:
                first_hits.append(result["first_hit"])
            unique_final.append(result["stats"][-1]["unique_chromosomes"])
        table.append({
            "success_count": len(first_hits),
            "success_rate": 100 * len(first_hits) / repetitions,
            "mean_best_fitness": statistics.mean(best_values),
            "mean_first_hit": statistics.mean(first_hits) if first_hits else None,
            "mean_final_diversity": statistics.mean(unique_final),
            "repetitions": repetitions,
        })
    return table


def compare_graph(results, output):
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, result in enumerate(results, 1):
        ax.plot([s["generation"] for s in result["stats"]],
                [s["max_fitness"] for s in result["stats"]],
                linewidth=1.8, label=f"Эксперимент {i}")
    ax.axhline(results[0]["optimum_fitness"], color="black", linewidth=1,
               linestyle="--", label="Дискретный максимум")
    ax.set(title="Сравнение лучших fitness по поколениям", xlabel="Поколение", ylabel="Fitness")
    ax.set_xlim(1, results[0]["params"].g)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def fmt(number, digits=6):
    return f"{number:.{digits}f}"


def build_report(results, repeat_summary, variant, base_folder):
    optimum_code, optimum_f = exact_discrete_optimum(variant)
    optimum_x = decode(optimum_code)
    g = results[0]["params"].g
    repeats = repeat_summary[0]["repetitions"]

    intro = f"""# Лабораторная работа №5

**Тема:** Реализация и исследование параметров генетического алгоритма  
**Вариант целевой функции:** №{variant}  
**Студент(ка):** ____________________  
**Группа:** ____________________  
**Преподаватель:** ____________________

## 1. Цель работы

Реализовать генетический алгоритм для поиска максимума функции одной переменной на отрезке [0, 1], изучить операторы селекции, кроссовера и мутации, а также исследовать влияние вероятностей `pc`, `pm` и размера популяции `N` на скорость и качество поиска.

## 2. Постановка задачи и целевая функция

Для варианта №{variant} необходимо максимизировать функцию

$$
{FORMULAS[variant]}, \\qquad x\\in[0,1].
$$

Алгоритм работает с **четырёхбитными** хромосомами (`L=4`). Поэтому он исследует только 16 допустимых точек: `0/15, 1/15, ..., 15/15`. Лучшее решение на этой сетке не обязательно совпадает с максимумом непрерывной функции на всём отрезке. Это ограничение точности кодирования, а не ошибка алгоритма.

**Контрольный точный перебор 16 хромосом** (независимая проверка результата ГА) даёт: хромосома **`{bits(optimum_code)}`**, десятичный код **{optimum_code}**, `x = {fmt(optimum_x)}`, `f(x) = {fmt(optimum_f)}`. Эта величина является *точным глобальным максимумом на заданной 4-битной сетке*.

## 3. Язык и технологии

- **Python 3.10+** — реализация алгоритма, случайная инициализация, вычисления, статистика и экспорт CSV.
- Модули стандартной библиотеки: `random` (генератор случайных чисел), `math` (целевая функция), `csv` (таблицы), `statistics` (средние значения), `argparse` (параметры запуска).
- **Matplotlib** — построение и сохранение графиков в PNG.
- Результаты: `README.md`, файлы подробных логов `full_log.txt`, таблицы `evolution.csv`, `statistics.csv`, графики `fitness.png` и сравнительный график `comparison.png`.

## 4. Представление решения и операторы

### 4.1. Кодирование и декодирование

Четыре бита кодируют целое число `b` от 0 до 15. Преобразование в вещественное значение:

$$
x = \\frac{{b}}{{2^L-1}} = \\frac{{b}}{{15}},\\qquad L=4.
$$

Например, `0101` → `b=5` → `x=5/15=0.333333`. Приспособленность (fitness) конкретной хромосомы равна значению целевой функции `f(x)`; **чем выше fitness, тем лучше**.

### 4.2. Основные этапы алгоритма

1. **Инициализация:** создаётся случайная популяция из `N` четырёхбитных хромосом. Повторяющиеся хромосомы допустимы.
2. **Оценка:** для каждой хромосомы вычисляются `b`, `x=b/15` и `fitness=f(x)`.
3. **Селекция:** *турнир размером 2* — случайно выбираются две особи; родителем становится особь с большим fitness. Подходит в том числе для функций с отрицательными значениями.
4. **Одноточечный кроссовер:** с вероятностью `pc` выбирается точка после 1-го, 2-го либо 3-го бита; родители обмениваются правыми частями. Иначе потомки являются копиями родителей.
5. **Побитовая мутация:** каждый бит каждого нового потомка независимо меняется `0 ↔ 1` с вероятностью `pm`. В логе указаны номер новой особи, позиция бита, старое и новое значения.
6. **Элитизм:** одна лучшая особь текущего поколения копируется в следующее **без мутации**. Это предохраняет найденный максимум от потери.
7. **Новое поколение:** элита плюс потомки заполняют ровно `N` мест. На последнем поколении создавать следующее не требуется.
8. **Остановка:** после заданных `G` поколений (`G ≥ 30`). Записываются наименьший, средний и наибольший fitness каждого поколения, а также лучшие найденные `x*`, `f(x*)`.

**Замечание:** из-за элитизма максимальный fitness поколения не уменьшается. Высокая мутация может увеличивать разнообразие, но разрушать хорошие комбинации у потомков; элита при этом сохраняется. Случайность может приводить к различным результатам даже при одинаковых параметрах.

## 5. Параметры проведённых экспериментов

Длина `L=4` фиксирована. Во всех основных опытах использован `G={g}`, фиксированное начальное значение генератора `seed={results[0]['params'].seed}` для воспроизводимости. При одинаковом `seed` первые четыре случайные хромосомы в стартовых популяциях совпадают.

"""
    params_table = markdown_table(
        ["№", "N", "L", "pc", "pm", "G", "Что проверяем"],
        [[1, 4, 4, 0.8, 0.05, g, "Базовые параметры"],
         [2, 4, 4, 0.4, 0.01, g, "Меньше скрещиваний и мутаций"],
         [3, 4, 4, 0.9, 0.15, g, "Больше скрещиваний и мутаций"],
         [4, 12, 4, 0.8, 0.05, g, "Влияние размера популяции (сравнение с №1)"]],
    )
    middle = f"""

Первое сравнение (№1–3) показывает влияние `pc` и `pm` при постоянном `N=4`. Сравнение №1 и №4 показывает влияние `N` при одинаковых `pc`, `pm` и `G`. Поскольку параметры `pc` и `pm` в №2–3 изменяются одновременно, по одиночным сравнениям нельзя строго отделить эффект одного от другого.

## 6. Результаты основных экспериментов

В итоговых таблицах указано лучшее решение, найденное **за все поколения**. Поколение первого обнаружения точного дискретного оптимума показывает скорость поиска; знак `—` означает, что оптимум за {g} поколений не встретился.

"""
    result_table = markdown_table(
        ["Эксп.", "Лучшая хромосома", "x*", "f(x*)", "Первое достижение оптимума", "Оптимум найден?"],
        [[i, f"`{bits(r['best_code'])}`", fmt(decode(r["best_code"])),
          fmt(r["best_fitness"]), str(r["first_hit"]) if r["first_hit"] is not None else "—",
          "Да" if math.isclose(r["best_fitness"], optimum_f, rel_tol=1e-12, abs_tol=1e-12) else "Нет"]
         for i, r in enumerate(results, 1)],
    )
    report = intro + params_table + middle + result_table
    report += "\n\n![Сравнение максимального fitness](results/comparison.png)\n\n"

    for i, result in enumerate(results, 1):
        p = result["params"]
        folder = f"results/experiment_{i}"
        hit = result["first_hit"]
        report += f"""### 6.{i}. Эксперимент №{i}

Параметры: `N={p.n}`, `L=4`, `pc={p.pc}`, `pm={p.pm}`, `G={p.g}`, `seed={p.seed}`. Найденная хромосома: **`{bits(result['best_code'])}`**; `x*={fmt(decode(result['best_code']))}`; `f(x*)={fmt(result['best_fitness'])}`. Первое обнаружение дискретного оптимума: **{f'поколение {hit}' if hit is not None else 'не обнаружен'}**.

![График fitness — эксперимент {i}]({folder}/fitness.png)

**Таблица статистики поколений:**

"""
        report += markdown_table(
            ["Поколение", "Минимальный fitness", "Средний fitness", "Максимальный fitness", "Разных хромосом"],
            [[s["generation"], fmt(s["min_fitness"]), fmt(s["mean_fitness"]),
              fmt(s["max_fitness"]), s["unique_chromosomes"]] for s in result["stats"]],
        )
        report += f"\n**Таблица эволюции популяции** — все {p.g * p.n} записей; также сохранена в [`evolution.csv`]({folder}/evolution.csv).\n\n"
        report += "<details>\n<summary>Показать полную таблицу особей по поколениям</summary>\n\n"
        report += markdown_table(
            ["Поколение", "№ особи", "Хромосома", "Код b", "x", "f(x)"],
            [[r["generation"], r["individual"], f"`{r['chromosome']}`", r["decimal"],
              fmt(r["x"]), fmt(r["fitness"])] for r in result["rows"]],
        )
        report += "\n</details>\n\n"
        report += (f"**Подробный автоматически сформированный лог всех {p.g} поколений:** "
                   f"[`full_log.txt`]({folder}/full_log.txt). Лог содержит все особи и fitness, "
                   "пары родителей, точки кроссовера, каждую мутацию, статистику и новые популяции.\n\n")
        report += "<details>\n<summary>Показать полный лог (включён в README для выполнения требований отчёта)</summary>\n\n```text\n"
        report += result["log"]
        report += "```\n\n</details>\n\n"

    report += f"""## 7. Статистическая проверка и анализ влияния параметров

Для более обоснованного сравнения каждый из четырёх наборов параметров был дополнительно запущен **{repeats} независимых раз**, с начальными случайными состояниями `300000, 300001, ...`. Для каждого набора используется один и тот же список seed. Таблица ниже рассчитана программой автоматически. *Доля успешных запусков* — это процент повторов, где обнаружен именно точный максимум среди всех 16 кодов. *Среднее поколение первого обнаружения* рассчитано **только по успешным повторам**.

"""
    report += markdown_table(
        ["Эксп.", "Успешно, запусков", "Успешность", "Средний лучший fitness", "Среднее поколение попадания*", "Разных хромосом в конце, среднее"],
        [[i, f"{s['success_count']} / {s['repetitions']}", f"{s['success_rate']:.1f}%",
          fmt(s["mean_best_fitness"]),
          f"{s['mean_first_hit']:.2f}" if s["mean_first_hit"] is not None else "—",
          f"{s['mean_final_diversity']:.2f}"] for i, s in enumerate(repeat_summary, 1)],
    )

    row1, row2, row3, row4 = repeat_summary
    report += f"""
*Сравнение основных параметров.* В опытах №1–3 популяция имеет одинаковый размер `N=4`; при уменьшении `pc` и `pm` изменяются интенсивности скрещивания и появления новых битовых комбинаций, а при увеличении обоих параметров повышается изменчивость потомков. Фактические доли обнаружения оптимума: №1 — **{row1['success_rate']:.1f}%**, №2 — **{row2['success_rate']:.1f}%**, №3 — **{row3['success_rate']:.1f}%**. Из этих опытов нельзя приписать разницу только одному из `pc` или `pm`, поскольку в №2 и №3 изменены сразу оба параметра.

*Влияние размера популяции.* №1 и №4 имеют одинаковые `pc=0.8`, `pm=0.05`, `G={g}`, но `N` увеличен с 4 до 12. По повторным запускам успешность составляет **{row1['success_rate']:.1f}%** против **{row4['success_rate']:.1f}%**. Более крупная популяция даёт больше кандидатов и может чаще находить удачные комбинации, но требует больше вычислений для каждого поколения (приблизительно пропорционально `N`).

*Ограничения анализа.* Поскольку длина хромосомы равна четырём битам, существует всего 16 различных кандидатов. Результаты характеризуют сходимость **на дискретной сетке**, а не точность непрерывного оптимума; стохастический алгоритм не гарантирует оптимум в каждом прогоне. Для строгой изоляции влияния `pc` и `pm` требуются дополнительные опыты, где меняется только один параметр.

## 8. Ответы на контрольные вопросы

**1. Что такое хромосома и ген?** Хромосома — закодированное решение, например `0101`. Ген — отдельный бит `0` или `1`. Здесь каждая хромосома содержит 4 гена.

**2. Как определяется fitness?** Двоичный код переводится в `b`, затем в `x=b/15`, после чего вычисляется целевая функция `f(x)`. Больший результат означает более подходящее решение.

**3. Для чего нужны селекция, кроссовер и мутация?** Селекция выбирает более перспективных родителей; кроссовер сочетает их части; мутация случайно меняет биты и помогает получать новые варианты.

**4. Как влияют pc и pm?** `pc` управляет вероятностью смешивания генов родителей, `pm` — вероятностью изменения каждого бита. Слишком малые значения могут замедлить поиск; слишком большая мутация может разрушать удачные решения.

**5. Что такое поколение?** Это набор `N` особей в определённый момент. Следующее поколение формируется из одной лучшей особи и новых потомков после селекции, кроссовера и мутации.

**6. Почему важно разнообразие?** При наличии разных хромосом алгоритм исследует больше областей поиска и реже застревает на одном неудачном решении.

**7. Как задаётся критерий останова?** Можно ограничить количество поколений, остановиться после достижения нужного fitness или если улучшений нет долгое время. Здесь используется строго `G ≥ 30` поколений.

**8. Как происходит декодирование?** Четырёхбитное число `b ∈ [0,15]` переводится в `x=b/(2^4-1)=b/15`. Например, `1100` = 12, значит `x=0.8`.

**9. Почему алгоритм может не найти глобальный максимум?** Отбор и мутация случайны, поколений ограниченное число, популяция может потерять разнообразие. Кроме того, четырёхбитное кодирование не позволяет проверить точки между узлами сетки.

**10. Как влияет N?** Увеличение популяции повышает разнообразие и может улучшить вероятность найти оптимум, но увеличивает количество вычислений на поколение.

**11. Для чего случайная инициализация?** Чтобы начать с разных решений и не ограничивать поиск одним заранее выбранным вариантом.

**12. Как повысить воспроизводимость?** Фиксировать `seed`, вариант функции, `N`, `L`, `pc`, `pm`, `G` и версию программы. При одинаковом seed и условиях получаются одинаковые результаты.

## 9. Вывод

В работе реализован генетический алгоритм с четырёхбитным кодированием, турнирной селекцией, одноточечным кроссовером, побитовой мутацией и элитизмом. Проведены четыре основных эксперимента (три по `pc`, `pm` и дополнительный по `N`) и статистическое сравнение на {repeats} повторениях. Для каждого опыта сохранены полные логи, таблицы эволюции, статистика по поколениям и график изменения fitness. Максимум на допустимой сетке при варианте №{variant} установлен точным перебором: **`{bits(optimum_code)}`, `x*={fmt(optimum_x)}`, `f(x*)={fmt(optimum_f)}`**. Влияние `pc` и `pm` проявляется в изменении активности поиска, а `N` — в разнообразии решений и вычислительной стоимости. Изменяя параметры, можно влиять на скорость и вероятность обнаружения лучшей хромосомы, но гарантии нахождения непрерывного глобального максимума нет.

## 10. Как запустить программу

Из папки с файлами:

```bash
pip install -r requirements.txt
python genetic_algorithm.py
```

Это заново создаст файлы результатов и `README.md`. Задать свою конфигурацию (значения `N` от 2 до 20, `G` не менее 30, `pc`/`pm` от 0 до 1):

```bash
python genetic_algorithm.py --single --variant {variant} --n 8 --pc 0.7 --pm 0.08 --g 40 --seed 123
```

Для смены индивидуального варианта во всех основных экспериментах:

```bash
python genetic_algorithm.py --variant 7
```

Программа поддерживает варианты функций 1–25; при смене варианта `README.md` и результаты пересчитываются. В режиме `--single` отчёт основных экспериментов не заменяется: вывод сохраняется в `results/custom_run/`.

## 11. Структура файлов

```text
lab5_genetic_algorithm/
├── README.md                 # полный отчёт, включая логи, таблицы и графики
├── genetic_algorithm.py      # исходный код, запуск и создание результатов
├── requirements.txt          # зависимость matplotlib
└── results/
    ├── comparison.png        # сравнение 4 основных экспериментов
    ├── repetitions.csv       # статистика по {repeats} повторениям
    ├── experiment_1/
    │   ├── full_log.txt      # все 30 поколений, операции и статистика
    │   ├── evolution.csv     # все особи каждого поколения
    │   ├── statistics.csv    # min/mean/max fitness по поколениям
    │   └── fitness.png       # график fitness
    ├── experiment_2/         # аналогичные 4 файла
    ├── experiment_3/         # аналогичные 4 файла
    └── experiment_4/         # аналогичные 4 файла, N = 12
```
"""
    (base_folder / "README.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Лабораторная №5: генетический алгоритм")
    parser.add_argument("--single", action="store_true", help="Один запуск с заданными параметрами")
    parser.add_argument("--variant", type=int, default=1, help="Номер функции от 1 до 25")
    parser.add_argument("--n", type=int, default=4, help="Размер популяции 2–20 (для --single)")
    parser.add_argument("--pc", type=float, default=0.8, help="Вероятность кроссовера 0–1 (для --single)")
    parser.add_argument("--pm", type=float, default=0.05, help="Вероятность мутации 0–1 (для --single)")
    parser.add_argument("--g", type=int, default=30, help="Количество поколений >= 30")
    parser.add_argument("--seed", type=int, default=154, help="Начальное значение генератора")
    parser.add_argument("--repetitions", type=int, default=100, help="Повторы для статистики")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    output = base / "results"
    params = Params(n=args.n, pc=args.pc, pm=args.pm, g=args.g, seed=args.seed, variant=args.variant)
    params.validate()

    if args.single:
        result = run_algorithm(params, detailed=True)
        export_experiment(result, output / "custom_run")
        print(f"Вариант №{args.variant}; лучшая хромосома: {bits(result['best_code'])}; "
              f"x={decode(result['best_code']):.6f}; f(x)={result['best_fitness']:.6f}")
        print("Файлы: results/custom_run/")
        return

    if args.repetitions < 1:
        parser.error("--repetitions должен быть не менее 1")
    experiments = [
        Params(4, 0.8, 0.05, args.g, args.seed, args.variant),
        Params(4, 0.4, 0.01, args.g, args.seed, args.variant),
        Params(4, 0.9, 0.15, args.g, args.seed, args.variant),
        Params(12, 0.8, 0.05, args.g, args.seed, args.variant),
    ]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for i, p in enumerate(experiments, 1):
        result = run_algorithm(p, detailed=True)
        results.append(result)
        export_experiment(result, output / f"experiment_{i}")
        print(f"Эксперимент {i}: {bits(result['best_code'])}, x={decode(result['best_code']):.6f}, "
              f"f(x)={result['best_fitness']:.6f}, первое попадание={result['first_hit']}")

    compare_graph(results, output / "comparison.png")
    repeated = repeat_experiments(experiments, args.repetitions, args.variant)
    write_csv(output / "repetitions.csv", [dict(experiment=i, **r) for i, r in enumerate(repeated, 1)],
              ["experiment", "repetitions", "success_count", "success_rate", "mean_best_fitness", "mean_first_hit", "mean_final_diversity"])
    build_report(results, repeated, args.variant, base)
    print("Готово: README.md, результаты экспериментов, логи, CSV и PNG.")


if __name__ == "__main__":
    main()
