# 该文件中实现了算法需要调用的一些通用函数
import random
import time
from helper.read_data import get_info_from_file

def test_feasiblity_and_cal_Cmax(p, s, sequence_1, sequence_2, machine_choices):
    """
    测试解是否可行（主要关注stage 6处是否出现有机器空闲情况下的等待），可行时计算目标函数值。

    :param p: 加工时间矩阵。
    :param s: 换模时间矩阵。
    :param sequence_1: 工位1-5上的加工次序，为数字1-n组成的list。
    :param sequence_2: 工位7-10上的加工次序，为数字1-n组成的list。
    :param machine_choices: 工件1，2...，n在工位6上选择的机器，含有n个0/1的list，0为较快的机器。
    :return: 目标函数值（若不可行，返回None）。
    """
    # 调整工件编号使之与索引一致
    sequence_1 = [0] + list(sequence_1)
    sequence_2 = [0] + list(sequence_2)
    machine_choices = [0] + list(machine_choices)

    C = [[0 for i in range(len(sequence_1))] for j in range(10+1)]    # C[i][j]表示工位i上工件j的完工时间

    for order in range(1, len(sequence_1)):
        for stage in range(1, 5+1):
            front = sequence_1[order-1]
            job = sequence_1[order]
            C[stage][job] = max(C[stage-1][job], C[stage][front]) + s[front][job] + p[stage][job]

    is_feasible = True
    front_in_m0 = 0
    front_in_m1 = 0
    for order in range(1, len(sequence_1)):
        job = sequence_1[order]
        if C[6][front_in_m0] > C[6][front_in_m1] and C[5][job] < C[6][front_in_m0]:
            # 此时只能选工位6中的第二台机器
            if machine_choices[job] == 0:
                is_feasible = False
        if C[6][front_in_m0] < C[6][front_in_m1] and C[5][job] < C[6][front_in_m1]:
            # 此时只能选工位6中的第一台机器
            if machine_choices[job] == 1:
                is_feasible = False
        if not is_feasible:
            # --------lines below are used for debug when not feasible ---------
            print(job)
            for line in C:
                print(line)
            # ------------------------------------------------------------------
            return None
        if machine_choices[job] == 0:
            C[6][job] = max(C[5][job], C[6][front_in_m0]) + s[front_in_m0][job] + p[6][job]
            front_in_m0 = job
        else:
            C[6][job] = max(C[5][job], C[6][front_in_m1]) + s[front_in_m1][job] + 1.2 * p[6][job]
            front_in_m1 = job

    for order in range(1, len(sequence_2)):
        for stage in range(7, 10 + 1):
            front = sequence_2[order - 1]
            job = sequence_2[order]
            C[stage][job] = max(C[stage - 1][job], C[stage][front]) + s[front][job] + p[stage][job]
    # ------------lines below are used for debug when feasible -----------------
    # for line in C:
    #     print(line)
    # --------------------------------------------------------------------------
    return max(C[10])


def generate_feasible_machine_choices_randomly(p, s, sequence_1):
    """
    为已经确定的工位1-5上的加工顺序给出一种可行（随机选择产生）的机器选择序列。

    :param p: 加工时间矩阵。
    :param s: 换模时间矩阵。
    :param sequence_1: 工位1-5上的工件加工顺序，为数字1-n组成的list。
    :return: 工件1，2...，n在工位6上选择的机器，含有n个0/1的list，0为较快的机器。
    """
    # 调整工件编号使之与索引一致
    sequence_1 = [0] + sequence_1

    machine_choices = [0 for i in range(len(sequence_1))]
    C = [[0 for i in range(len(sequence_1))] for j in range(6 + 1)]  # C[i][j]表示工位i上工件j的完工时间
    for order in range(1, len(sequence_1)):
        for stage in range(1, 5+1):
            front = sequence_1[order-1]
            job = sequence_1[order]
            C[stage][job] = max(C[stage-1][job], C[stage][front]) + s[front][job] + p[stage][job]

    front_in_stage_6 = [0, 0]
    for order in range(1, len(sequence_1)):
        job = sequence_1[order]
        if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[0]]:
            # 此时只能选工位6中的第二台机器
            machine_choices[job] = 1
            C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + s[front_in_stage_6[1]][job] + 1.2 * p[6][job]
            front_in_stage_6[1] = job
        elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[1]]:
            # 此时只能选工位6中的第一台机器
            machine_choices[job] = 0
            C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + s[front_in_stage_6[0]][job] + p[6][job]
            front_in_stage_6[0] = job
        else:
            m = random.randint(0, 1)
            machine_choices[job] = m
            C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + s[front_in_stage_6[m]][job] + (1 + m * 0.2) * p[6][job]
            front_in_stage_6[m] = job
    del machine_choices[0]
    return machine_choices


def generate_a_feasible_solution(p, s, num_of_jobs):
    """
    随机生成一个可行解。

    :param p: 加工时间矩阵。
    :param s: 换模时间矩阵。
    :param num_of_jobs: 工件总数。
    :return: 工位1-5上加工顺序，工位7-10上加工顺序，工位6上机器选择序列。
    """
    random_sequence_1 = [i for i in range(1, num_of_jobs + 1)]
    random.shuffle(random_sequence_1)
    random_sequence_2 = [i for i in range(1, num_of_jobs + 1)]
    random.shuffle(random_sequence_2)
    feasible_machine_choices = generate_feasible_machine_choices_randomly(p, s, random_sequence_1)
    return random_sequence_1, random_sequence_2, feasible_machine_choices


def get_feasible_choices(p, s, n, feasible_choices_list, C, seq1, choices, choices_len, front_in_stage_6):
    """递归函数，注意该函数仅供list_all_feasible_machine_choices函数调用，是一个辅助函数"""
    for order in range(choices_len + 1, n + 1):
        job = seq1[order]
        if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[0]]:
            # 此时只能选工位6中的第二台机器
            choices[job] = 1
            C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + s[front_in_stage_6[1]][job] + 1.2 * p[6][job]
            front_in_stage_6[1] = job
            choices_len += 1
            if choices_len == n:
                feasible_choices_list.append(choices[1:])
                return
        elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[1]]:
            # 此时只能选工位6中的第一台机器
            choices[job] = 0
            C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + s[front_in_stage_6[0]][job] + p[6][job]
            front_in_stage_6[0] = job
            choices_len += 1
            if choices_len == n:
                feasible_choices_list.append(choices[1:])
                return
        else:
            for m in range(2):
                front_in_stage_6_copy = front_in_stage_6[:]
                choices[job] = m
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + s[front_in_stage_6[m]][job] + (1 + m * 0.2) * p[6][job]
                front_in_stage_6_copy[m] = job
                if choices_len + 1 == n:
                    feasible_choices_list.append(choices[1:])
                else:
                    get_feasible_choices(p, s, n, feasible_choices_list, C, seq1, choices, choices_len + 1, front_in_stage_6_copy)
            return


def list_all_feasible_machine_choices(p, s, sequence_1, C=None):
    """
    为给定的sequence_1枚举出所有可行的机器选择序列。
    
    :param p: 加工时间矩阵。
    :param s: 换模时间矩阵。
    :param sequence_1: 工位1-5上的工件加工顺序，为数字1-n组成的list。
    :param C: 待计算的完工时间矩阵（可选）。
    :return: 所有可行的机器选择序列构成的列表。
    """
    n = len(s) - 1
    # 调整工件编号使之与索引一致
    sequence_1 = [0] + sequence_1

    if C is None:
        C = [[0 for i in range(len(sequence_1))] for j in range(6 + 1)]  # C[i][j]表示工位i上工件j的完工时间
    for order in range(1, len(sequence_1)):
        for stage in range(1, 5 + 1):
            front = sequence_1[order - 1]
            job = sequence_1[order]
            C[stage][job] = max(C[stage-1][job], C[stage][front]) + s[front][job] + p[stage][job]
    feasible_choices_list = []
    get_feasible_choices(p, s, n, feasible_choices_list, C, sequence_1, [0 for i in range(n+1)], 0, [0, 0])
    return feasible_choices_list


def get_best_machine_choices(p, s, sequence_1, sequence_2, C=None, feasible_choices_list=None):
    """
    通过测试可以发现可行的机器序列规模实际上很小，对可行机器序列进行枚举，选择能够使得工位1-5
    及工位7-10上加工顺序固定时，总加工用时最短的机器序列。（通过性能测试，每个算例平均用时约
    0.001s）

    :param p: 加工时间矩阵。
    :param s: 换模时间矩阵。
    :param sequence_1: 工位1-5上的工件加工顺序，为数字1-n组成的list。
    :param sequence_2: 工位7-10上的工件加工顺序，为数字1-n组成的list。
    :param C: 待计算的完工时间矩阵。
    :param feasible_choices_list: 可行的机器选择序列构成的list。
    :return: sequence_1, sequence_2固定情况下最优的机器选择序列。
    """
    n = len(s) - 1
    # 调整工件编号使之与索引一致
    sequence_1 = [0] + sequence_1

    if C is None:
        C = [[0 for i in range(n+1)] for j in range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间
        for order in range(1, n+1):
            for stage in range(1, 5 + 1):
                front = sequence_1[order - 1]
                job = sequence_1[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + s[front][job] + p[stage][job]
        feasible_choices_list = []
        get_feasible_choices(p, s, n, feasible_choices_list, C, sequence_1,
                             [0 for i in range(n + 1)], 0, [0, 0])
    # 寻找最优的机器选择序列
    sequence_2 = [0] + sequence_2
    min_Cmax = 1000000
    best_m_choices = []
    for m_choices in feasible_choices_list:
        skip = False
        front_in_stage_6 = [0, 0]
        for order in range(1, n + 1):
            job = sequence_1[order]
            m_choice = m_choices[job - 1]
            front = front_in_stage_6[m_choice]
            C[6][job] = max(C[5][job], C[6][front]) + s[front][job] + (1 + m_choice * 0.2) * p[6][job]
            front_in_stage_6[m_choice] = job
        Cmax = 0
        for order in range(1, n + 1):
            for stage in range(7, 10 + 1):
                front = sequence_2[order - 1]
                job = sequence_2[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + s[front][job] + p[stage][job]
                Cmax = C[stage][job]
                if Cmax > min_Cmax:
                    skip = True
                    break
            if skip:
                break
        if skip:
            continue
        if Cmax < min_Cmax:
            min_Cmax = Cmax
            best_m_choices = m_choices
    return best_m_choices


def get_matrix_of_begin_time_and_end_time(seq1, seq2, m_choices):
    """
    根据工位1-5上加工顺序，工位7-10上加工顺序，工位6上机器选择情况计算完工时间矩阵和开工时间矩阵。

    :param seq1: 工位1-5上加工顺序。
    :param seq2: 工位7-10上加工顺序。
    :param m_choices: 工位6上机器选择序列。
    :return: 开工时间矩阵，完工时间矩阵。
    """
    n = len(seq1)
    C = [[0 for i in range(n+1)] for j in range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间
    B = [[0 for i in range(n+1)] for j in range(10 + 1)]  # B[i][j]表示工位i上工件j的开工时间

    sequence_1 = [0] + seq1
    sequence_2 = [0] + seq2
    machine_choices = [0] + m_choices

    for order in range(1, len(sequence_1)):
        for stage in range(1, 5+1):
            front = sequence_1[order-1]
            job = sequence_1[order]
            C[stage][job] = max(C[stage-1][job], C[stage][front]) + s[front][job] + p[stage][job]

    front_in_m0 = 0
    front_in_m1 = 0
    for order in range(1, len(sequence_1)):
        job = sequence_1[order]
        if machine_choices[job] == 0:
            C[6][job] = max(C[5][job], C[6][front_in_m0]) + s[front_in_m0][job] + p[6][job]
            front_in_m0 = job
        else:
            C[6][job] = max(C[5][job], C[6][front_in_m1]) + s[front_in_m1][job] + 1.2 * p[6][job]
            front_in_m1 = job

    for order in range(1, len(sequence_2)):
        for stage in range(7, 10 + 1):
            front = sequence_2[order - 1]
            job = sequence_2[order]
            C[stage][job] = max(C[stage - 1][job], C[stage][front]) + s[front][job] + p[stage][job]

    for stage in range(1, 5 + 1):
        for job in range(1, n+1):
            front = sequence_1[sequence_1.index(job) - 1]
            B[stage][job] = max(C[stage-1][job], C[stage][front])

    seq_on_machine_0 = [0]
    seq_on_machine_1 = [0]
    for i in range(n):
        job = seq1[i]
        if machine_choices[job] == 0:
            seq_on_machine_0.append(job)
        else:
            seq_on_machine_1.append(job)
    for job in range(1, n+1):
        if job in seq_on_machine_0:
            front = seq_on_machine_0[seq_on_machine_0.index(job) - 1]
        else:
            front = seq_on_machine_1[seq_on_machine_1.index(job) - 1]
        B[6][job] = max(C[5][job], C[6][front])

    for stage in range(7, 10 + 1):
        for job in range(1, n + 1):
            front = sequence_2[sequence_2.index(job) - 1]
            B[stage][job] = max(C[stage - 1][job], C[stage][front])

    return B, C


def verify_triangle_inequality(p, s):
    # 证明加工时间是否存在“三角不等式”
    from itertools import permutations
    n = len(s) - 1
    for jobs in permutations([i + 1 for i in range(n)], 3):
        i = jobs[0]
        j = jobs[1]
        k = jobs[2]
        for stage in range(1, 10 + 1):
            if s[i][k] > s[i][j] + p[stage][j] + s[j][k]:
                print(i, j, k)


def get_lower_bound(p, s, permutation_len=1):
    # 结合机器视角和工件视角的下界计算方法
    from itertools import permutations
    n = len(s) - 1
    lower_bounds = []
    modified_p = [[0 for i in range(n+1)] for j in range(10 + 1)]
    for i in range(1, 10+1):
        for j in range(1, n+1):
            modified_p[i][j] = p[i][j] + sorted([s[k][j] for k in range(1, n + 1)])[1]
    for stage in range(1, 10+1):
        print(stage)
        if stage == 6:
            continue
        makespans = []
        for seq in list(permutations([i + 1 for i in range(n)], permutation_len)):
            first_job = seq[0]
            min_time_before = sum([p[i][first_job] for i in range(stage)])
            makespan = min_time_before + p[stage][first_job]
            for i in range(n):
                if i + 1 in seq:
                    continue
                makespan += modified_p[stage][i+1]
            for i in range(permutation_len):
                if i == 0:
                    continue
                makespan += s[seq[i-1]][seq[i]] + p[stage][seq[i]]
            min_time_after = 1000000
            for i in range(n):
                if i + 1 in seq:
                    continue
                time_after = sum([modified_p[s][i+1] for s in range(stage + 1, 10 + 1)])
                if time_after < min_time_after:
                    min_time_after = time_after
            makespan += min_time_after
            # print("stage: ", stage, "\tseq: ", seq, "\tmakespan: ", makespan)
            makespans.append(makespan)
        lower_bounds.append(min(makespans))
    return max(lower_bounds)


def get_lower_bound2(p, s, permutation_len=1):
    """
    结合机器视角和工件视角的下界计算方法，与get_lower_bound的唯一区别是主循环的内外层级交换了，这两种方式
    获取的下界可能有所不同，但都为该问题下界。对case 2，该方法获取的下界更优一些。
    """
    from itertools import permutations
    n = len(s) - 1
    lower_bounds = []
    modified_p = [[0 for i in range(n+1)] for j in range(10 + 1)]
    for i in range(1, 10+1):
        for j in range(1, n+1):
            modified_p[i][j] = p[i][j] + sorted([s[k][j] for k in range(1, n + 1)])[1]
    for seq in list(permutations([i + 1 for i in range(n)], permutation_len)):
        makespans = []
        for stage in range(1, 10 + 1):
            if stage == 6:
                continue
            first_job = seq[0]
            min_time_before = sum([p[i][first_job] for i in range(stage)])
            makespan = min_time_before + p[stage][first_job]
            for i in range(n):
                if i + 1 in seq:
                    continue
                makespan += modified_p[stage][i+1]
            for i in range(permutation_len):
                if i == 0:
                    continue
                makespan += s[seq[i-1]][seq[i]] + p[stage][seq[i]]
            min_time_after = 1000000
            for i in range(n):
                # if i + 1 in seq:
                #     continue
                time_after = sum([modified_p[s][i+1] for s in range(stage + 1, 10 + 1)])
                if time_after < min_time_after:
                    min_time_after = time_after
            makespan += min_time_after
            # print("stage: ", stage, "\tseq: ", seq, "\tmakespan: ", makespan)
            makespans.append(makespan)
        lower_bounds.append(max(makespans))
    return min(lower_bounds)


if __name__ == '__main__':
    p, s = get_info_from_file('../data/case 2.xlsx')
    # for i in range(1000):
    #     s1, s2, c = generate_a_feasible_solution(len(s) - 1)
    #     assessment = test_feasiblity_and_cal_Cmax(p, s, s1, s2, c)
    #     print(s1, s2, c)
    #     print(assessment)
    # s1, s2, c = SPTCH(p, s).solve()
    # print(assessment)

    # 测试get_best_machine_choices函数时间性能
    # start_time = time.time()
    # iter_num = 1000
    # for i in range(iter_num):
    #     print(i)
    #     lst1 = [i + 1 for i in range(n)]
    #     random.shuffle(lst1)
    #     feasible_choices = list_all_feasible_machine_choices(p, s, lst1)
    #     lst2 = [i + 1 for i in range(n)]
    #     random.shuffle(lst2)
    #     m_choices = get_best_machine_choices(p, s, lst1, lst2)
    # end_time = time.time()
    # print('Avg Time: ', (end_time - start_time) / iter_num)

    # 测试get_machine_choices_via_heuristic函数时间性能
    # start_time = time.time()
    # iter_num = 1000
    # count = 0
    # for i in range(iter_num):
    #     print(i)
    #     lst1 = [i + 1 for i in range(n)]
    #     random.shuffle(lst1)
    #     lst2 = [i + 1 for i in range(n)]
    #     random.shuffle(lst2)
    #     m_choices1 = get_machine_choices_via_heuristic(p, s, lst1, lst2)
    #     print(m_choices1)
    #     m_choices2 = get_best_machine_choices(p, s, lst1, lst2)
    #     print(m_choices2)
    #     if m_choices1 == m_choices2:
    #         count += 1
    # end_time = time.time()
    # print('Avg Time: ', (end_time - start_time) / iter_num)
    # print(count)

    # 测试get_best_machine_choices函数正确性
    # for i in range(10):
    #     lst = [i + 1 for i in range(n)]
    #     random.shuffle(lst)
    #     feasible_choices = list_all_feasible_machine_choices(p, s, lst)
    #     lst2 = [i + 1 for i in range(n)]
    #     random.shuffle(lst2)
    #     min_Cmax = 1000000
    #     best_m_choices = []
    #     for choices in feasible_choices:
    #         result = test_feasiblity_and_cal_Cmax(p, s, lst, lst2, choices)
    #         if result < min_Cmax:
    #             min_Cmax = result
    #             best_m_choices = choices
    #     if best_m_choices == get_best_machine_choices(p, s, lst, lst2):
    #         print('True')
    #     else:
    #         print('False')
    #     print('')

    # case 2下界求解
    start_time = time.time()
    print(get_lower_bound2(p, s, 3))
    print(time.time() - start_time)
