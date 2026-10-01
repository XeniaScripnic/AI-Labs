import time

DEPTH = 5
WIDTH = 2

LEAVES = [
    9, 8, 7, 6, 5, 4, 3, 2,
    8, 7, 6, 5, 4, 3, 2, 1,
    7, 6, 5, 4, 3, 2, 1, 0,
    6, 5, 4, 3, 2, 1, 0, -1
]

minimax_nodes = 0
alpha_beta_nodes = 0
leaf_index = 0


# Создание бинарного дерева
def create_tree(depth):
    global leaf_index

    if depth == 0:
        value = LEAVES[leaf_index]
        leaf_index += 1
        return value

    return [create_tree(depth - 1) for _ in range(WIDTH)]


# Обычный Minimax
def minimax(node, maximizing):
    global minimax_nodes
    minimax_nodes += 1

    if not isinstance(node, list):
        return node

    values = [minimax(child, not maximizing) for child in node]

    if maximizing:
        return max(values)
    else:
        return min(values)


# Minimax с Alpha-Beta отсечением
def alpha_beta(node, alpha, beta, maximizing):
    global alpha_beta_nodes
    alpha_beta_nodes += 1

    if not isinstance(node, list):
        return node

    if maximizing:
        result = -float("inf")

        for child in node:
            result = max(
                result,
                alpha_beta(child, alpha, beta, False)
            )

            alpha = max(alpha, result)

            if alpha >= beta:
                break

        return result

    else:
        result = float("inf")

        for child in node:
            result = min(
                result,
                alpha_beta(child, alpha, beta, True)
            )

            beta = min(beta, result)

            if alpha >= beta:
                break

        return result


tree = create_tree(DEPTH)

# Обычный Minimax
start = time.perf_counter()
minimax_result = minimax(tree, True)
minimax_time = time.perf_counter() - start

# Minimax с Alpha-Beta
start = time.perf_counter()
alpha_beta_result = alpha_beta(
    tree,
    -float("inf"),
    float("inf"),
    True
)
alpha_beta_time = time.perf_counter() - start

# Вывод результатов
print("MINIMAX + ALPHA-BETA")

print("\nГлубина дерева:", DEPTH)
print("Ширина дерева:", WIDTH)

print("\nMINIMAX:")
print("Результат:", minimax_result)
print("Проверено узлов:", minimax_nodes)
print("Время:", minimax_time)

print("\nALPHA-BETA:")
print("Результат:", alpha_beta_result)
print("Проверено узлов:", alpha_beta_nodes)
print("Время:", alpha_beta_time)

print("\nСравнение:")
print("Экономия узлов:", minimax_nodes - alpha_beta_nodes)
