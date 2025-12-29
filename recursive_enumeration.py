# 该文件采用递归枚举的算法求解问题
from helper.read_data import get_info_from_file
import time


def completion_time_recursion(state, C, seq1, seq2, choices, choices_len, front_in_stage_6, solutions, best_sol):
    """
    用于实现for循环嵌套层数可变。
    注意：如果只是想要通过best_sol的第一个元素设置搜索上限，需要将state == 3情况下最优解的更新
    语句注释掉。

    :param state: 当前所在状态，0，1，2，3分别表示在决定1-5工位上加工顺序、在决定工位6上机器选择、
        在决定7-10工位上加工顺序、对最优解进行更新（如需要）。
    :param C: 完工时间矩阵。
    :param seq1: 1-5工位上加工顺序。
    :param seq2: 7-10工位上加工顺序。
    :param choices: 工位6上机器的选择。
    :param choices_len: 当前工位6上已选好机器的工件数量。
    :param front_in_stage_6: 工位6上当前各台机器上上一个完工的工件号。
    :param solutions: 当前所有可行的解。
    :param best_sol: 一个四元组，元素分别为目标函数值，seq1，seq2，choices
    """
    if state == 3:  # 获取整体完工时间
        obj = max(C[10])
        if obj <= best_sol[0]:
            new_solution = (obj, seq1, seq2, choices[:])
            solutions.append(new_solution)
            # add comment for the four lines below if necessary
            best_sol[0] = obj
            best_sol[1] = seq1
            best_sol[2] = seq2
            best_sol[3] = choices[:]
            print(new_solution)
        return

    if state == 1:    # 工位6上的机器尚未选择
        for order in range(choices_len+1, n+1):
            job = seq1[order]
            if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[0]]:
                # 此时只能选工位6中的第二台机器
                choices[job] = 1
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + s[front_in_stage_6[1]][job] + 1.2 * p[6][job]
                if C[6][job] > best_sol[0]:
                    return
                front_in_stage_6[1] = job
                choices_len += 1
                if choices_len == n:
                    completion_time_recursion(2, C, seq1, [0],
                                              choices, choices_len,
                                              front_in_stage_6,
                                              solutions, best_sol)
            elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[1]]:
                # 此时只能选工位6中的第一台机器
                choices[job] = 0
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + s[front_in_stage_6[0]][job] + p[6][job]
                if C[6][job] > best_sol[0]:
                    return
                front_in_stage_6[0] = job
                choices_len += 1
                if choices_len == n:
                    completion_time_recursion(2, C, seq1, [0],
                                              choices, choices_len,
                                              front_in_stage_6,
                                              solutions, best_sol)
            else:
                for m in range(2):
                    front_in_stage_6_copy = front_in_stage_6[:]
                    choices[job] = m
                    C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + s[front_in_stage_6[m]][job] + (1 + m * 0.2) * p[6][job]
                    front_in_stage_6_copy[m] = job
                    if C[6][job] > best_sol[0]:
                        return
                    if choices_len + 1 == n:
                        state = 2
                    completion_time_recursion(state, C, seq1, [0], choices, choices_len + 1,
                                              front_in_stage_6_copy,
                                              solutions, best_sol)
                return

    if state == 0: # 工位1-5上顺序暂未确定
        for job in range(1, n+1):
            if job not in seq1:
                for k in range(1, 5+1):
                    front = seq1[-1]
                    C[k][job] = max(C[k-1][job], C[k][front]) + s[front][job] + p[k][job]
                    if C[k][job] > best_sol[0]:  # 不选择这个job
                        return
                if len(seq1) + 1 == n+1:
                    print(seq1 + [job])
                    completion_time_recursion(1, C, seq1+[job], [0], choices, 0, [0, 0], solutions, best_sol)
                else:
                    completion_time_recursion(0, C, seq1+[job], [0], choices, 0, [0, 0], solutions, best_sol)
        return

    if state == 2:  # 7-10工位上顺序暂未确定
        # 对序列2无任何先验知识的情况
        for job in range(1, n+1):
            if job not in seq2:
                better_sol_existed = True
                for k in range(7, 10+1):
                    front = seq2[-1]
                    C[k][job] = max(C[k-1][job], C[k][front]) + s[front][job] + p[k][job]
                    if C[k][job] > best_sol[0]:  # 不选择这个job
                        better_sol_existed = False
                        break
                if not better_sol_existed:
                    continue
                if len(seq2) + 1 == n + 1:
                    completion_time_recursion(3, C, seq1, seq2+[job], choices, choices_len, front_in_stage_6, solutions, best_sol)
                else:
                    completion_time_recursion(2, C, seq1, seq2+[job], choices, choices_len, front_in_stage_6, solutions, best_sol)
        return


if __name__ == "__main__":
    p, s = get_info_from_file('data/case 1.xlsx', print_table=True)
    n = len(s) - 1

    start = time.time()
    C = [[0 for i in range(n+1)] for j in range(10+1)]    # C[i][j]表示工位i上工件j的完工时间
    solutions = []
    # best_solution的第一个元素可以输入当前已知的问题上界，加速剪枝和搜索
    best_solution = [10000, [], [], []]
    completion_time_recursion(0, C, [0], [0], [0 for i in range(n+1)], 0, [0, 0], solutions, best_solution)
    # print(len(solutions))
    print(time.time() - start)
    print('best:')
    print(best_solution)
