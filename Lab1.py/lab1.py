def check_string(s):
    i = 0
    n = 0
    m = 0

    # Считаем символы "a"
    while i < len(s) and s[i] == "a":
        n += 1
        i += 1

    # После "a" считаем символы "b"
    while i < len(s) and s[i] == "b":
        m += 1
        i += 1

    # Строка должна быть обработана полностью
    # и содержать хотя бы один символ a или b
    if i == len(s) and n + m >= 1:
        return True
    else:
        return False


while True:
    s = input("Введите строку: ")

    # 0 используется для завершения программы
    if s == "0":
        print("Программа завершена.")
        break

    if check_string(s):
        print("Строка принадлежит языку.")
    else:
        print("Строка не принадлежит языку.")