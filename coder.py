from math import log, factorial

# Global var init
bin_code_comb = []


def dec_num_to_bin_list(num):
    return list(bin(num))[2:]


def xor(first_row, second_row):
    result_row = []
    for i in range(0, len(first_row)):
        if first_row[i] == second_row[i]:
            result_row.append(0)
        else:
            result_row.append(1)

    return result_row


def xor_multiple_rows(matrix):
    if len(matrix) == 0:
        return [0]
    if len(matrix) == 1:
        return matrix[0]

    buffer_row = []
    for i in range(0, len(matrix[0])):
        buffer_row.append(0)

    for i in range(0, len(matrix)):
        buffer_row = xor(buffer_row, matrix[i])

    return buffer_row


def recursive_vertor_division_shanfano(vector):
    if len(vector) == 1:
        return vector

    counter = 0
    sum_elems = 0
    sum_1 = 0
    sum_2 = 0
    vector_match = 0

    for i in range(0, len(vector)):
        vector_match = vector_match + vector[i][0]
    vector_match /= 2

    while True:
        sum_elems += vector[counter][0]

        if sum_elems >= vector_match:
            sum_1 = recursive_vertor_division_shanfano(vector[: counter + 1])
            sum_2 = recursive_vertor_division_shanfano(vector[counter + 1 :])
            break

        counter += 1

    for item in sum_1:
        bin_code_comb[item[1]].append(1)
    for item in sum_2:
        bin_code_comb[item[1]].append(0)

    return sum_1 + sum_2


def shannon_fano(vector):
    vector.sort()
    vector.reverse()
    vector_len = len(vector)

    vector_new = []
    for i in range(0, len(vector)):
        vector_new.append([vector[i], i])
    vector = vector_new

    global bin_code_comb
    bin_code_comb = []
    for i in range(len(vector)):
        bin_code_comb.append([])

    recursive_vertor_division_shanfano(vector)

    avg_len = 0
    for i in range(0, len(bin_code_comb)):
        bin_code_comb[i] = bin_code_comb[i][::-1]
        avg_len += len(bin_code_comb[i])

        print(vector[i][0], bin_code_comb[i])
    return avg_len / len(vector)


def huffman(vector):
    vector_prim = vector.copy()
    vector.sort()
    vector.reverse()

    vector_new = []
    for i in range(0, len(vector)):
        vector_new.append([vector[i], i])
    vector = vector_new

    global bin_code_comb
    bin_code_comb = []
    for i in range(len(vector)):
        bin_code_comb.append([])

    minimal_1 = []
    minimal_1_index = 0
    minimal_2 = []
    minimal_2_index = 0
    while len(vector) != 1:
        minimal_1 = [2, 0]
        minimal_1_index = 1000
        minimal_2 = [2, 0]
        minimal_2_index = 1000

        for i in range(0, len(vector)):
            if vector[i][0] < minimal_1[0]:
                minimal_1 = vector[i]
                minimal_1_index = i
        vector.pop(minimal_1_index)

        for i in range(0, len(vector)):
            if vector[i][0] < minimal_2[0]:
                minimal_2 = vector[i]
                minimal_2_index = i
        vector.pop(minimal_2_index)

        if minimal_1[0] == minimal_2[0]:
            minimal_1, minimal_2 = minimal_2, minimal_1

        min_1_type = type(minimal_1[1])
        min_2_type = type(minimal_2[1])

        if isinstance(minimal_1[1], list) and isinstance(minimal_2[1], list):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), minimal_1[1] + minimal_2[1]]
            )

            for item in minimal_1[1]:
                bin_code_comb[item].append(0)
            for item in minimal_2[1]:
                bin_code_comb[item].append(1)

        if isinstance(minimal_1[1], list) and isinstance(minimal_2[1], int):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), minimal_1[1] + [minimal_2[1]]]
            )

            for item in minimal_1[1]:
                bin_code_comb[item].append(0)
            bin_code_comb[minimal_2[1]].append(1)

        if isinstance(minimal_1[1], int) and isinstance(minimal_2[1], list):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), [minimal_1[1]] + minimal_2[1]]
            )

            bin_code_comb[minimal_1[1]].append(0)
            for item in minimal_2[1]:
                bin_code_comb[item].append(1)

        if isinstance(minimal_1[1], int) and isinstance(minimal_2[1], int):
            vector.append(
                [round(minimal_1[0] + minimal_2[0], 2), [minimal_1[1]] + [minimal_2[1]]]
            )

            bin_code_comb[minimal_1[1]].append(0)
            bin_code_comb[minimal_2[1]].append(1)

    avg_len = 0
    for i in range(0, len(bin_code_comb)):
        bin_code_comb[i] = bin_code_comb[i][::-1]
        avg_len += len(bin_code_comb[i])
        print(vector_prim[i], bin_code_comb[i])
    return avg_len / len(vector_prim)


def first_processing(matrix):
    print("-" * 40)
    print("Task 1")
    print("Matrix G")
    for item in matrix:
        print(item)
    print()
    # Calculation of n, k, r
    n = len(matrix[0])
    k = len(matrix)
    r = n - k

    # Calculation of weight coefficients
    bin_counter_vec = 0

    w = []
    for i in range(0, len(matrix[0])):
        w.append(0)

    for i in range(0, 2**k):
        bin_counter_vec = dec_num_to_bin_list(i)

        for i in range(len(dec_num_to_bin_list(2**k - 1)) - len(bin_counter_vec)):
            bin_counter_vec.insert(0, "0")

        print(bin_counter_vec, end="\t")

        rows_to_xor = []
        for j in range(0, len(bin_counter_vec)):
            if bin_counter_vec[j] == "1":
                rows_to_xor.append(matrix[j])

        xor_result = xor_multiple_rows(rows_to_xor)
        print(xor_result, end=" ")

        ones_counter = 0
        for elem in xor_result:
            if elem == 1:
                ones_counter += 1

        print(ones_counter)

        for j in range(0, len(w)):
            if ones_counter == j:
                w[j] += 1

    # Calculation d, qo, qi
    for i, num in enumerate(w[1:]):
        if num != 0:
            d = i + 1
            break
    qo = d - 1
    qi = round(qo / 2)

    # Results
    print(f"\nn = {n}")
    print(f"k = {k}")
    print(f"r = {r}")
    print(f"d = {d}")
    print(f"qo = {qo}")
    print(f"qi = {qi}")
    print(f"w = {w}")


def fourth_processing(n, k, p, system_base=2):
    print("\n" + "-" * 40)
    print("Task 4")
    print(f"n = {n}")
    print(f"k = {k}")
    print(f"p = {p}")

    r = n - k
    print(f"\n2**r = {system_base**r}")
    N_sindr = (system_base**r) - 1
    print(f"N sindr = {N_sindr}\n")

    # q max calculation
    qi = 1
    prev_sum = 0
    while True:

        summ = 0
        for q in range(1, qi + 1):
            summ += (factorial(n)) / (factorial(q) * factorial(n - q))
        print(f"if qi = {qi}, then sum[Cqn] = {summ}")
        if summ > N_sindr:
            qi -= 1
            print(f"\nsum[Cqn] bigger than 2^r - 1")
            print(f"{summ} > {N_sindr}")
            print(f"then 1 step back, qi = {qi}, sum[Cqn] = {prev_sum}")
            break
        qi += 1
        prev_sum = summ

    dk = 2 * qi + 1

    # P err decoding calculation
    summ = 0
    for q in range(qi + 1, n + 1):
        summ += (
            ((factorial(n)) / (factorial(q) * factorial(n - q)))
            * (p**q)
            * ((1 - p) ** (n - q))
        )
    print(f"P err dec = {summ}")
    print(f"P bit dec = {summ*((dk)/(n))}")


def seventh_processing(vector: list, system_base=3):
    print("\n" + "-" * 40)
    print("Task 7")
    print("Vector", vector)
    print("Standart code R =", round(log(len(vector), system_base), 2))

    print("\nHuffman code")
    print("R =", round(huffman(vector), 2))

    print("\nShannon-Fano code")
    print("R =", round(shannon_fano(vector), 2))

    counter = 0
    while counter < len(vector):
        vector[counter] = round(vector[counter] + vector[counter + 1], 2)
        vector.pop(counter + 1)
        counter += 1

    print("\nNew vector", vector)
    print("Standart code R =", round(log(len(vector), system_base), 2))
    print("\nHuffman code")
    print("R =", round(huffman(vector), 2))

    print("\nShannon-Fano code")
    print("R =", round(shannon_fano(vector), 2))


if __name__ == "__main__":

    with open("Data.txt") as file:
        file_data = file.readlines()

        # Read task 1
        G = []
        counter = 0
        for i in range(file_data.index("# 1\n") + 1, file_data.index("# 4\n")):
            current_line = list(file_data[i])
            G.append([])
            for j in range(0, len(current_line)):
                if j % 2 == 0:
                    G[counter].append(int(current_line[j]))
            counter += 1

        # Read task 4
        for i in range(file_data.index("# 4\n") + 1, file_data.index("# 7\n")):
            current_line = file_data[i]
            n_tsk_4 = int(current_line[2 : current_line.index("k") - 1])
            k_tsk_4 = int(
                current_line[current_line.index("k") + 2 : current_line.index("p") - 1]
            )
            p_tsk_4 = float(current_line[current_line.index("p") + 2 : -1])

        # Read task 7
        vector_tsk_7 = ""
        for i in range(file_data.index("# 7\n") + 1, len(file_data)):
            current_line = file_data[i]
            vector_tsk_7 = current_line.split()
            for j in range(0, len(vector_tsk_7)):
                vector_tsk_7[j] = float(vector_tsk_7[j])

    first_processing(G)
    fourth_processing(n_tsk_4, k_tsk_4, p_tsk_4)
    # fourth_processing(24, 13, 0.071964)
    seventh_processing(vector_tsk_7, 2)
