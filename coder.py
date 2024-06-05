from matrix_operations import *


def Task_1(matrix):
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

    # w alt
    w_dict = {}
    for index, elem in enumerate(w):
        if elem != 0:
            w_dict[index] = elem

    # Results
    print(f"\nn = {n}")
    print(f"k = {k}")
    print(f"r = n - k = {r}")
    print(f"d = min w = {d}")
    print(f"qo = d - 1 = {qo}")
    print(f"qi = (d - 1) / 2 = {qi}")
    print(f"w = {w}")
    print(f"w = {w_dict}")


def Task_2(primary_matrix: list, operations_to_solve=10):
    print("\n" + "-" * 40)
    print("\nTask 2\nMatrix G")

    matrix_print(primary_matrix)

    matrix = deepcopy(primary_matrix)

    columns = len(matrix[0])
    rows = len(matrix)

    if get_single_num_columns_index(matrix)[0] < rows:
        print("Matrix does not have enough columnns with single 1")
        matrix = matrix_to_single_num_columns(matrix, operations_to_solve)
        print("New matrix")
        matrix_print(matrix)
    else:
        print("Matrix have enough columnns with single 1\n")

    matrix_with_ones = deepcopy(matrix)

    matrix = matrix_to_unit_matrix(matrix)
    print("G sist [IQ]")
    matrix_print(matrix)

    p = matrix_columns_swap_index(matrix_with_ones, matrix)

    print("Pi")
    for swap in p:
        print(swap[0] + 1, end=" ")
    print()
    for swap in p:
        print(swap[1] + 1, end=" ")

    I = div_matrix_in_two(matrix, div_index=rows - 1)[0]
    Q = div_matrix_in_two(matrix, div_index=rows - 1)[1]

    print("\n\nQ transposed")
    matrix_print(matrix_transpose(Q))

    print("H sist [Q^T, I]")
    H_sist = join_matrix_in_one(matrix_transpose(Q), I)
    matrix_print(H_sist)

    p_inverted = sorted(p, key=lambda x: x[1])
    for elem in range(0, len(p_inverted)):
        p_inverted[elem][0], p_inverted[elem][1] = (
            p_inverted[elem][1],
            p_inverted[elem][0],
        )

    print("Pi^-1")
    for swap in p_inverted:
        print(swap[0] + 1, end=" ")
    print()
    for swap in p_inverted:
        print(swap[1] + 1, end=" ")

    H = matrix_columns_swap_by_list(H_sist, p_inverted)
    print("\n\nH")
    matrix_print(H)

    print("Checking the calculated matrix H, by H*G transposed")
    matrix_print(matrix_accuracy_HG(H, primary_matrix, print_messages=True))


def Task_4(n, k, p, system_base=2):
    print("\n" + "-" * 40)
    print("Task 4")
    print(f"n = {n}")
    print(f"k = {k}")
    print(f"p = {p}")

    r = n - k
    print(f"\n2**r = {system_base**r}")
    N_sindr = (system_base**r) - 1
    print(f"N sindr = 2**r - 1 = {N_sindr}\n")

    # q max calculation
    print("qi, is max value at which sum(Cqn) is lower or equal then 2**r")
    qi = 1
    prev_sum = 0
    while True:

        summ = 0
        for q in range(1, qi + 1):
            summ += (factorial(n)) / (factorial(q) * factorial(n - q))
        print(f"if qi = {qi}, then sum(Cqn) = {summ}")
        if summ > N_sindr:
            qi -= 1
            print(f"\nsum(Cqn) now bigger than 2^r - 1")
            print(f"{summ} > {N_sindr}")
            print(f"then 1 step back. qi = {qi}, sum(Cqn) = {prev_sum}")
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


def Task_7(vector: list, system_base=3):
    print("\n" + "-" * 40)
    print("Task 7")
    print("Vector", vector)
    print(
        f"Standart code R = log({system_base}){len(vector)} = ",
        round(log(len(vector), system_base), 2),
    )

    print("\nHuffman code")
    print("R = avg bit len =", round(huffman(vector), 2))

    print("\nShannon-Fano code")
    print("R = avg bit len =", round(shannon_fano(vector), 2))

    print("\nVector P(X) turns into vector P(XX)")
    new_vector = []
    for p_i in vector:
        for p_j in vector:
            new_vector.append(round(p_i * p_j, 3))

    print("\nNew vector", new_vector)
    print(
        f"Standart code R = log({system_base}){len(new_vector)} = ",
        round(log(len(new_vector), system_base), 2),
    )

    print("\nHuffman code")
    print("R = avg bit len =", round(huffman(new_vector), 2))

    print("\nShannon-Fano code")
    print("R = avg bit len =", round(shannon_fano(new_vector), 2))


if __name__ == "__main__":

    with open("Data.txt") as file:
        file_data = file.readlines()

        # Read task 1
        G_1 = []
        counter = 0
        for i in range(file_data.index("# 1\n") + 1, file_data.index("# 2\n")):
            current_line = list(file_data[i])
            G_1.append([])
            for j in range(0, len(current_line)):
                if j % 2 == 0:
                    G_1[counter].append(int(current_line[j]))
            counter += 1

        # Read task 2
        G_2 = []
        counter = 0
        for i in range(file_data.index("# 2\n") + 1, file_data.index("# 4\n")):
            current_line = list(file_data[i])
            G_2.append([])
            for j in range(0, len(current_line)):
                if j % 2 == 0:
                    G_2[counter].append(int(current_line[j]))
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

    Task_1(G_1)
    Task_2(G_2, 4)
    Task_4(n_tsk_4, k_tsk_4, p_tsk_4)
    Task_7(vector_tsk_7, 3)
