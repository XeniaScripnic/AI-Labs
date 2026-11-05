"""Лабораторная работа №4, вариант №1: нечёткий прогноз осадков (Мамдани).

Запуск окна:               python mamdani_weather.py
Проведение экспериментов:  python mamdani_weather.py --tests

Внешние зависимости: numpy, matplotlib. Tkinter входит в стандартную библиотеку Python.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# 1. ПАРАМЕТРЫ МОДЕЛИ — их можно менять независимо от алгоритма.
DOMAINS = {
    "temperature": (-20.0, 45.0),  # °C
    "humidity": (0.0, 100.0),      # %
    "pressure": (950.0, 1050.0),   # гПа
    "rain": (0.0, 100.0),          # %, условная оценка вероятности
}

# Для каждого терма: (a, b, c, d) трапециевидной функции принадлежности.
# При b == c она становится треугольной, при a == b / c == d — плечевой.
TERMS = {
    "temperature": {
        "низкая": (-20, -20, 0, 15),
        "умеренная": (5, 20, 20, 35),
        "высокая": (25, 35, 45, 45),
    },
    "humidity": {
        "низкая": (0, 0, 25, 45),
        "средняя": (30, 50, 50, 70),
        "высокая": (55, 75, 100, 100),
    },
    "pressure": {
        "низкое": (950, 950, 980, 1000),
        "среднее": (990, 1010, 1010, 1030),
        "высокое": (1020, 1040, 1050, 1050),
    },
    "rain": {
        "низкая": (0, 0, 20, 45),
        "средняя": (30, 50, 50, 70),
        "высокая": (55, 80, 100, 100),
    },
}

DISPLAY_NAMES = {
    "temperature": "Температура (°C)",
    "humidity": "Влажность (%)",
    "pressure": "Давление (гПа)",
    "rain": "Вероятность осадков (%)",
}


@dataclass(frozen=True)
class Rule:
    """Условия: (имя переменной, терм); соединение: AND/OR."""
    conditions: tuple[tuple[str, str], ...]
    operator: str
    outcome: str
    description: str


# 2. ПРАВИЛА — первые 9 соответствуют примерам варианта из задания.
# Дополнительные правила 10–12 обеспечивают вывод для любого значения влажности.
# Правило 13 демонстрирует OR и уточняет ситуацию для холодной погоды.
RULES = [
    Rule((("temperature", "низкая"), ("humidity", "высокая")), "AND", "высокая", "Низкая температура И высокая влажность"),
    Rule((("temperature", "умеренная"), ("humidity", "высокая")), "AND", "высокая", "Умеренная температура И высокая влажность"),
    Rule((("temperature", "высокая"), ("pressure", "высокое")), "AND", "низкая", "Высокая температура И высокое давление"),
    Rule((("temperature", "высокая"), ("humidity", "низкая")), "AND", "низкая", "Высокая температура И низкая влажность"),
    Rule((("pressure", "низкое"), ("humidity", "высокая")), "AND", "высокая", "Низкое давление И высокая влажность"),
    Rule((("pressure", "среднее"), ("humidity", "средняя")), "AND", "средняя", "Среднее давление И средняя влажность"),
    Rule((("temperature", "низкая"), ("pressure", "низкое")), "AND", "высокая", "Низкая температура И низкое давление"),
    Rule((("temperature", "умеренная"), ("pressure", "высокое")), "AND", "низкая", "Умеренная температура И высокое давление"),
    Rule((("temperature", "высокая"), ("humidity", "средняя")), "AND", "средняя", "Высокая температура И средняя влажность"),
    Rule((("humidity", "низкая"),), "AND", "низкая", "Низкая влажность (дополнительное правило)"),
    Rule((("humidity", "средняя"),), "AND", "средняя", "Средняя влажность (дополнительное правило)"),
    Rule((("humidity", "высокая"),), "AND", "высокая", "Высокая влажность (дополнительное правило)"),
    Rule((("temperature", "низкая"), ("humidity", "высокая"), ("pressure", "низкое")), "OR", "высокая", "Низкая температура ИЛИ высокая влажность ИЛИ низкое давление"),
]

# Для компактной демонстрации OR все условия правила 13 объединяются через ИЛИ.
# В произвольном приложении возможны более сложные составные логические выражения.

# Сетка с шагом 0.1 процента для численного вычисления центроида.
OUTPUT_X = np.linspace(*DOMAINS["rain"], 1001)


def trapezoid(x: float | np.ndarray, a: float, b: float, c: float, d: float):
    """Функция принадлежности: от 0 до 1, корректна для плечевых термов."""
    arr = np.asarray(x, dtype=float)
    left = np.ones_like(arr) if a == b else np.clip((arr - a) / (b - a), 0, 1)
    right = np.ones_like(arr) if c == d else np.clip((d - arr) / (d - c), 0, 1)
    result = np.minimum(left, right)
    result = np.where((arr < a) | (arr > d), 0.0, result)
    if np.ndim(arr) == 0:
        return float(result)
    return result


def membership(variable: str, term: str, x: float | np.ndarray):
    return trapezoid(x, *TERMS[variable][term])


def validate_input(values: dict[str, float]) -> None:
    """Понятная проверка границ и числового формата."""
    for key in ("temperature", "humidity", "pressure"):
        if key not in values:
            raise ValueError(f"Не указано значение: {DISPLAY_NAMES[key]}.")
        try:
            number = float(values[key])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Поле «{DISPLAY_NAMES[key]}» должно содержать число.") from exc
        lo, hi = DOMAINS[key]
        if not math.isfinite(number):
            raise ValueError(f"Поле «{DISPLAY_NAMES[key]}» должно содержать конечное число.")
        if not lo <= number <= hi:
            raise ValueError(f"Для «{DISPLAY_NAMES[key]}» допустим диапазон от {lo:g} до {hi:g}.")


@dataclass
class Result:
    value: float
    category: str
    recommendation: str
    degrees: dict[str, dict[str, float]]
    strengths: list[float]
    aggregated: np.ndarray


def calculate(values: dict[str, float]) -> Result:
    """Мамдани: фаззификация -> правила -> агрегация -> центроид."""
    validate_input(values)

    # 3. ФАЗЗИФИКАЦИЯ: степени принадлежности каждой входной переменной.
    degrees = {
        name: {term: membership(name, term, float(values[name])) for term in TERMS[name]}
        for name in ("temperature", "humidity", "pressure")
    }

    # 4. НЕЧЁТКИЕ ПРАВИЛА: AND = min, OR = max.
    strengths: list[float] = []
    aggregated = np.zeros_like(OUTPUT_X)
    for rule in RULES:
        parts = [degrees[name][term] for name, term in rule.conditions]
        strength = min(parts) if rule.operator == "AND" else max(parts)
        strengths.append(strength)

        # Импликация Мамдани: срезаем выходное множество на уровне активации.
        clipped_output = np.minimum(strength, membership("rain", rule.outcome, OUTPUT_X))
        # Агрегация: объединяем выходы правил операцией max.
        aggregated = np.maximum(aggregated, clipped_output)

    # 5. ДЕФАЗЗИФИКАЦИЯ: центроид агрегированной функции принадлежности.
    area = float(np.trapezoid(aggregated, OUTPUT_X))
    if area <= 0:
        raise RuntimeError("Нет активированных правил: проверьте функции принадлежности.")
    value = float(np.trapezoid(OUTPUT_X * aggregated, OUTPUT_X) / area)
    if value < 35:
        category = "Низкая"
        recommendation = "По модели вероятность осадков низкая."
    elif value < 65:
        category = "Средняя"
        recommendation = "По модели вероятность осадков умеренная."
    else:
        category = "Высокая"
        recommendation = "По модели вероятность осадков высокая."

    return Result(value, category, recommendation, degrees, strengths, aggregated)


# 6. ТЕСТОВЫЕ СЦЕНАРИИ: во всех тестах одинаковая база правил.
SCENARIOS = [
    ("Холодно и сыро", {"temperature": 2, "humidity": 90, "pressure": 975}),
    ("Жарко и сухо", {"temperature": 37, "humidity": 20, "pressure": 1045}),
    ("Умеренные условия", {"temperature": 20, "humidity": 50, "pressure": 1010}),
]


def plot_memberships(fig, values: dict[str, float], result: Result):
    """Четыре графика: три входа и итоговая нечёткая функция."""
    fig.clear()
    axes = fig.subplots(2, 2).ravel()
    colors = ["#2463ad", "#e6a02c", "#259b78"]
    for ax, name in zip(axes[:3], ("temperature", "humidity", "pressure")):
        xs = np.linspace(*DOMAINS[name], 350)
        for (term, params), color in zip(TERMS[name].items(), colors):
            ax.plot(xs, trapezoid(xs, *params), color=color, label=term, linewidth=2)
        ax.axvline(values[name], color="#bd3636", linestyle="--", linewidth=1.5,
                   label=f"Ввод: {values[name]:g}")
        ax.set(xlabel=DISPLAY_NAMES[name], ylabel="Степень μ", ylim=(-0.03, 1.1))
        ax.set_title(DISPLAY_NAMES[name], fontsize=11)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)

    ax = axes[3]
    for (term, params), color in zip(TERMS["rain"].items(), colors):
        ax.plot(OUTPUT_X, trapezoid(OUTPUT_X, *params), color=color,
                label=term, alpha=0.55, linewidth=1.6)
    ax.fill_between(OUTPUT_X, result.aggregated, color="#6c77a3", alpha=0.42,
                    label="Агрегированный вывод")
    ax.plot(OUTPUT_X, result.aggregated, color="#4a5282", linewidth=1.8)
    ax.axvline(result.value, color="#bd3636", linestyle="--", linewidth=1.8,
               label=f"Центроид: {result.value:.2f}%")
    ax.set(xlabel="Вероятность осадков (%)", ylabel="Степень μ", ylim=(-0.03, 1.1))
    ax.set_title("Выход и дефаззификация", fontsize=11)
    ax.grid(alpha=0.18)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(pad=2.3)


def run_tests() -> None:
    """Проверка трёх сценариев + границ + некорректного ввода и сохранение файлов."""
    from matplotlib.figure import Figure
    root = Path(__file__).resolve().parent
    results = []
    for title, vals in SCENARIOS:
        res = calculate(vals)
        assert math.isfinite(res.value) and 0 <= res.value <= 100
        assert any(s > 0 for s in res.strengths)
        results.append((title, vals, res))
        print(f"{title:20} | T={vals['temperature']:5.1f} | H={vals['humidity']:5.1f} | "
              f"P={vals['pressure']:6.1f} | {res.value:6.2f}% | {res.category}")

    # Отдельно проверяем крайние допустимые значения и ошибки.
    for vals in (
        {"temperature": -20, "humidity": 0, "pressure": 950},
        {"temperature": 45, "humidity": 100, "pressure": 1050},
    ):
        assert math.isfinite(calculate(vals).value)
    for vals in (
        {"temperature": "", "humidity": 50, "pressure": 1010},
        {"temperature": float("nan"), "humidity": 50, "pressure": 1010},
        {"temperature": 50, "humidity": 50, "pressure": 1010},
        {"temperature": 20, "humidity": 101, "pressure": 1010},
        {"temperature": 20, "humidity": 50},
    ):
        try:
            calculate(vals)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Программа приняла некорректный ввод: {vals}")

    with (root / "test_scenarios.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Сценарий", "Температура, °C", "Влажность, %", "Давление, гПа", "Результат, %", "Категория"])
        for title, vals, res in results:
            writer.writerow([title, vals["temperature"], vals["humidity"], vals["pressure"], f"{res.value:.2f}", res.category])

    # Этот PNG можно вставить в README; строится без открытия графического окна.
    fig = Figure(figsize=(11, 7.5), dpi=130)
    plot_memberships(fig, SCENARIOS[2][1], results[2][2])
    fig.savefig(root / "membership_functions.png", bbox_inches="tight")
    for i, (title, vals, res) in enumerate(results, 1):
        scenario_fig = Figure(figsize=(10, 6.7), dpi=115)
        plot_memberships(scenario_fig, vals, res)
        scenario_fig.savefig(root / f"scenario_{i}.png", bbox_inches="tight")
    print("\nПроверка успешно пройдена: 3 сценария, граничные значения и ошибки ввода.")
    print("Сохранены test_scenarios.csv, membership_functions.png и scenario_1.png ... scenario_3.png")


def start_gui() -> None:
    """Интерактивное окно: ввод, расчёт, повторные тесты и графики."""
    import tkinter as tk
    from tkinter import messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Лабораторная №4 — вероятность осадков | алгоритм Мамдани")
    root.geometry("1240x800")
    root.minsize(1050, 680)
    root.configure(bg="#f5f7fa")

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("TButton", padding=6)
    style.configure("TLabel", font=("Segoe UI", 10))

    content = ttk.Frame(root, padding=15)
    content.pack(expand=True, fill="both")
    content.columnconfigure(1, weight=1)
    content.rowconfigure(0, weight=1)
    left = ttk.Frame(content, width=325)
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
    right = ttk.Frame(content)
    right.grid(row=0, column=1, sticky="nsew")

    ttk.Label(left, text="Прогноз вероятности осадков", font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(0, 4))
    ttk.Label(left, text="Нечёткая модель Мамдани · 13 правил", foreground="#515a68").pack(anchor="w", pady=(0, 14))

    inputs = {}
    defaults = {"temperature": "20", "humidity": "50", "pressure": "1010"}
    for name in ("temperature", "humidity", "pressure"):
        lo, hi = DOMAINS[name]
        ttk.Label(left, text=f"{DISPLAY_NAMES[name]}  [{lo:g} ... {hi:g}]").pack(anchor="w", pady=(7, 3))
        var = tk.StringVar(value=defaults[name])
        ttk.Entry(left, textvariable=var, width=35, font=("Segoe UI", 11)).pack(fill="x")
        inputs[name] = var

    value_label = ttk.Label(left, text="", font=("Segoe UI", 17, "bold"), foreground="#215a99")
    category_label = ttk.Label(left, text="", font=("Segoe UI", 11, "bold"))
    recommendation_label = ttk.Label(left, text="", wraplength=315)

    rules_header = ttk.Label(left, text="Активированные правила:", font=("Segoe UI", 10, "bold"))
    rules_box = tk.Text(left, height=11, width=39, wrap="word", font=("Segoe UI", 9),
                        bg="#ffffff", relief="solid", borderwidth=1)
    rules_box.configure(state="disabled")

    fig = Figure(figsize=(9.1, 7), dpi=100)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    def do_calculate():
        try:
            raw = {name: var.get().strip().replace(",", ".") for name, var in inputs.items()}
            result = calculate(raw)
        except ValueError as err:
            messagebox.showerror("Некорректные данные", str(err), parent=root)
            return
        except RuntimeError as err:
            messagebox.showerror("Ошибка расчёта", str(err), parent=root)
            return

        values = {name: float(raw[name]) for name in raw}
        value_label.config(text=f"Результат: {result.value:.2f}%")
        category_label.config(text=f"Категория: {result.category}")
        recommendation_label.config(text=result.recommendation)
        rules_box.configure(state="normal")
        rules_box.delete("1.0", "end")
        active = [(i + 1, r, s) for i, (r, s) in enumerate(zip(RULES, result.strengths)) if s > 1e-9]
        for number, rule, strength in sorted(active, key=lambda item: -item[2]):
            rules_box.insert("end", f"№{number} (μ={strength:.2f}): {rule.description} → {rule.outcome}\n\n")
        rules_box.configure(state="disabled")
        plot_memberships(fig, values, result)
        canvas.draw_idle()

    def fill_values(vals: dict[str, float]):
        for name, number in vals.items():
            inputs[name].set(str(number))
        do_calculate()

    ttk.Button(left, text="Рассчитать", command=do_calculate).pack(fill="x", pady=(15, 4))
    ttk.Button(left, text="Сбросить значения", command=lambda: fill_values({
        "temperature": 20, "humidity": 50, "pressure": 1010
    })).pack(fill="x", pady=(0, 10))
    ttk.Label(left, text="Готовые тестовые сценарии:").pack(anchor="w", pady=(3, 4))
    for label, vals in SCENARIOS:
        ttk.Button(left, text=label, command=lambda v=vals: fill_values(v)).pack(fill="x", pady=2)
    value_label.pack(anchor="w", pady=(12, 4))
    category_label.pack(anchor="w", pady=(0, 4))
    recommendation_label.pack(anchor="w", pady=(0, 12))
    rules_header.pack(anchor="w", pady=(0, 4))
    rules_box.pack(fill="both", expand=True)
    ttk.Label(left, text="Учебная модель, не реальный метеопрогноз.", foreground="#6e7480").pack(anchor="w", pady=(7, 0))
    root.bind("<Return>", lambda _event: do_calculate())
    do_calculate()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Лабораторная №4: нечёткая оценка осадков (Мамдани)")
    parser.add_argument("--tests", action="store_true", help="Запустить 3 теста и создать PNG/CSV")
    args = parser.parse_args()
    if args.tests:
        run_tests()
    else:
        start_gui()


if __name__ == "__main__":
    main()
