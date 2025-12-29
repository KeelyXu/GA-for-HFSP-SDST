# 该文件实现了一些构造型启发式，可作为元启发式的初始解
import itertools

class SPTCH:
    """
    SPT Cyclic Heuristic
    【由于工位1-5上加工顺序一致，工位7-10上加工顺序一致，该规则被大大简化】

    工件在stage 1上按照修正后的加工时间（工件自身在stage 1上的加工时间与该工件和下一加工
    工件之间最小换模时间之和）的升序进行加工顺序的排序。
    在stage 6上，如果能够选择机器，为工件i选择使其在stage 6上完工时间最短的机器。
    在stage 7上，根据stage 6上的完工时间的升序进行加工顺序的排序（即先入先出）。

    注意：该方法实际上并未利用stage 6, 7之间的缓冲区。

    参考文献：Scheduling flexible flow lines with sequence-dependent setup times.
    """
    def __init__(self, pt, st):
        self.p = pt
        self.s = st
        self.job_num = len(st) - 1

    def solve(self):
        # 先确定工位1-5上加工顺序
        modified_process_t = []
        for job_id in range(1, self.job_num + 1):
            modified_p = self.p[1][job_id] + sorted(self.s[job_id])[2]
            modified_process_t.append(modified_p)
        sequence_1 = [i[0] + 1 for i in sorted(enumerate(modified_process_t), key=lambda x: x[1])]
        sequence_1.insert(0, 0)     # 调整工件编号使之与索引一致

        C = [[0 for i in range(len(sequence_1))] for j in range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间

        for order in range(1, len(sequence_1)):
            for stage in range(1, 5 + 1):
                front = sequence_1[order - 1]
                job = sequence_1[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + self.s[front][job] + self.p[stage][job]

        # 再确定工位6上的机器选择
        machine_choices = [0 for i in range(len(sequence_1))]

        front_in_stage_6 = [0, 0]
        for order in range(1, len(sequence_1)):
            job = sequence_1[order]
            if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[0]]:
                # 此时只能选工位6中的第二台机器
                machine_choices[job] = 1
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + self.s[front_in_stage_6[1]][job] + 1.2 * self.p[6][job]
                front_in_stage_6[1] = job
            elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[1]]:
                # 此时只能选工位6中的第一台机器
                machine_choices[job] = 0
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + \
                            self.s[front_in_stage_6[0]][job] + self.p[6][job]
                front_in_stage_6[0] = job
            else:
                t0 = self.s[front_in_stage_6[0]][job] + self.p[6][job]
                t1 = self.s[front_in_stage_6[1]][job] + self.p[6][job] * 1.2
                m = 0 if t0 < t1 else 1
                machine_choices[job] = m
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + \
                            self.s[front_in_stage_6[m]][job] + \
                            (1+m*0.2) * self.p[6][job]
                front_in_stage_6[m] = job
        complete_time_in_stage_6 = C[6]
        # 最后确定工位7-10上加工顺序
        sequence_2 = [c[0] for c in sorted(enumerate(complete_time_in_stage_6), key=lambda x: x[1])]
        return sequence_1[1:], sequence_2[1:], machine_choices[1:]


class FTMIH:
    """
    FTMIH是一种最小化各阶段流程时间和的多次插入启发式，是求解TSP的插入算法在多机器、多阶段
    问题上的改造版本。setup time通过修正后的加工时间得以体现。
    【由于工位1-5上加工顺序一致，工位7-10上加工顺序一致，且工位6前不允许等待，该规则被大大简化】

    参考文献：Scheduling flexible flow lines with sequence-dependent setup times.
    """
    def __init__(self, pt, st):
        self.p = pt
        self.s = st
        self.job_num = len(st) - 1

    def solve(self):
        # 先确定工位1-5上的加工顺序
        modified_process_t = [(self.p[1][job_id] + sorted(self.s[job_id])[2]) for job_id in range(1, self.job_num+1)]
        job_order = [i[0] + 1 for i in sorted(enumerate(modified_process_t), key=lambda x: x[1])]
        sequence_1 = [0]
        for job_id in job_order:
            possible_positions = [i for i in range(1, len(sequence_1)+1)]
            best_sum = 1000000
            best_pos = None
            for possible_pos in possible_positions:
                sequence_1.insert(possible_pos, job_id)
                completion_t = [0.0]
                for i in range(1, len(sequence_1)):
                    job = sequence_1[i]
                    front = sequence_1[i-1]
                    completion_time = completion_t[i-1] + self.s[front][job] + self.p[1][job]
                    completion_t.append(completion_time)
                if sum(completion_t) < best_sum:
                    best_sum = sum(completion_t)
                    best_pos = possible_pos
                sequence_1.remove(job_id)
            sequence_1.insert(best_pos, job_id)

        C = [[0 for i in range(len(sequence_1))] for j in range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间

        for order in range(1, len(sequence_1)):
            for stage in range(1, 5 + 1):
                front = sequence_1[order - 1]
                job = sequence_1[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + self.s[front][job] + self.p[stage][job]

        # 再确定工位6上的机器选择
        machine_choices = [0 for i in range(len(sequence_1))]
        front_in_stage_6 = [0, 0]
        for order in range(1, len(sequence_1)):
            job = sequence_1[order]
            if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][
                job] < C[6][front_in_stage_6[0]]:
                # 此时只能选工位6中的第二台机器
                machine_choices[job] = 1
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + self.s[front_in_stage_6[1]][job] + 1.2 * self.p[6][job]
                front_in_stage_6[1] = job
            elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][job] < C[6][front_in_stage_6[1]]:
                # 此时只能选工位6中的第一台机器
                machine_choices[job] = 0
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + self.s[front_in_stage_6[0]][job] + self.p[6][job]
                front_in_stage_6[0] = job
            else:
                t0 = self.s[front_in_stage_6[0]][job] + self.p[6][job]
                t1 = self.s[front_in_stage_6[1]][job] + self.p[6][job] * 1.2
                m = 0 if t0 < t1 else 1
                machine_choices[job] = m
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + self.s[front_in_stage_6[m]][job] + (1 + m * 0.2) * self.p[6][job]
                front_in_stage_6[m] = job
        complete_time_in_stage_6 = C[6]

        # 最后确定工位7-10上加工顺序
        ready_times = complete_time_in_stage_6
        modified_process_t = [(self.p[7][job_id] + sorted(self.s[job_id])[2]) for job_id in range(1, self.job_num + 1)]
        job_order = [i[0] + 1 for i in sorted(enumerate(modified_process_t), key=lambda x: x[1])]
        sequence_2 = [0]
        for job_id in job_order:
            possible_positions = [i for i in range(1, len(sequence_2)+1)]
            best_sum = 1000000
            best_pos = None
            for possible_pos in possible_positions:
                sequence_2.insert(possible_pos, job_id)
                completion_t = [0.0]
                for i in range(1, len(sequence_2)):
                    job = sequence_2[i]
                    front = sequence_2[i-1]
                    completion_time = max(completion_t[i-1], C[6][job]) + self.s[front][job] + self.p[7][job]
                    completion_t.append(completion_time)
                if sum(completion_t) - sum(ready_times[:len(completion_t)]) < best_sum:
                    best_sum = sum(completion_t) - sum(ready_times[:len(completion_t)])
                    best_pos = possible_pos
                sequence_2.remove(job_id)
            sequence_2.insert(best_pos, job_id)

        return sequence_1[1:], sequence_2[1:], machine_choices[1:]


class JohnsonsRule:
    """
    g/2, g/2 Johnson's rule是对Johnson's rule的扩展，考虑了超过两阶段的流水车间的setup
    time。（g表示总阶段数）

    参考文献：Scheduling flexible flow lines with sequence-dependent setup times.
    """
    def __init__(self, pt, st):
        self.p = pt
        self.s = st
        self.job_num = len(st) - 1

    def solve(self):
        # 对所有工件，计算阶段1~g/2及g/2~g阶段上的修正加工时间和
        U_with_p1 = []
        V_with_pg = []
        for i in range(self.job_num):
            job = i + 1
            p1 = sum([(self.p[stage][job] + sorted(self.s[job])[2]) for stage in range(1, 5+1)])
            pg = sum([(self.p[stage][job] + sorted(self.s[job])[2]) for stage in range(5+1, 10+1)])
            if p1 < pg:
                U_with_p1.append((job, p1))
            else:
                V_with_pg.append((job, pg))
        U_with_p1.sort(key=lambda x:x[1])
        U = [element[0] for element in U_with_p1]
        V_with_pg.sort(key=lambda x:x[1], reverse=True)
        V = [element[0] for element in V_with_pg]
        U.extend(V)
        # 确定工位1-5上的加工顺序
        sequence_1 = U
        sequence_1.insert(0, 0)  # 调整工件编号使之与索引一致

        C = [[0 for i in range(len(sequence_1))] for j in
             range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间

        for order in range(1, len(sequence_1)):
            for stage in range(1, 5 + 1):
                front = sequence_1[order - 1]
                job = sequence_1[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + self.s[front][job] + self.p[stage][job]

        # 再确定工位6上的机器选择
        machine_choices = [0 for i in range(len(sequence_1))]

        front_in_stage_6 = [0, 0]
        for order in range(1, len(sequence_1)):
            job = sequence_1[order]
            if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][
                job] < C[6][front_in_stage_6[0]]:
                # 此时只能选工位6中的第二台机器
                machine_choices[job] = 1
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + self.s[front_in_stage_6[1]][job] + 1.2 * self.p[6][job]
                front_in_stage_6[1] = job
            elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][
                job] < C[6][front_in_stage_6[1]]:
                # 此时只能选工位6中的第一台机器
                machine_choices[job] = 0
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + self.s[front_in_stage_6[0]][job] + self.p[6][job]
                front_in_stage_6[0] = job
            else:
                t0 = self.s[front_in_stage_6[0]][job] + self.p[6][job]
                t1 = self.s[front_in_stage_6[1]][job] + self.p[6][job] * 1.2
                m = 0 if t0 < t1 else 1
                machine_choices[job] = m
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + self.s[front_in_stage_6[m]][job] + (1 + m * 0.2) * self.p[6][job]
                front_in_stage_6[m] = job
        complete_time_in_stage_6 = C[6]
        # 最后确定工位7-10上加工顺序
        sequence_2 = [c[0] for c in sorted(enumerate(complete_time_in_stage_6), key=lambda x: x[1])]
        return sequence_1[1:], sequence_2[1:], machine_choices[1:]


class FRB4_1:
    """
        FRB算法是NEH算法的一种有效变种，相比于NEH，在搜索阶段进行了更多的插入尝试；在FBR算法中，
    FBR4在计算时间增加不显著的情况下对结果有较好的提升，FBR4的主要思想是在插入1个工件后对其
    插入位置前后的p个工件进行重新插入，这样的算法被记作FBR_p算法，考虑到计算开销，p通常取1，
    即这里的FBR4_1算法。
        考虑到我们处理的问题并不是permutation flow shop问题，只对sequence 1用该算法。在机
    器选择阶段，仍采用最早完工原则选择机器，并以工位6上的完工顺序作为工位7-10上的加工顺序。

    参考文献：Constraint based local search for flowshops with sequence-dependent
    setup times.
    """
    def __init__(self, pt, st):
        self.p = pt
        self.s = st
        self.job_num = len(st) - 1

    def cal_Cmax(self, seq):
        seq = [0] + seq
        C = [[0 for i in range(self.job_num + 1)] for j in range(5 + 1)]
        Cmax = 0
        for order in range(1, len(seq)):
            for stage in range(1, 5 + 1):
                front = seq[order - 1]
                job = seq[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + self.s[front][job] + self.p[stage][job]
                Cmax = C[stage][job]
        return Cmax

    def solve(self):
        A = []
        for j in range(self.job_num):
            # 用1-5工位还是1-10工位上的加工时间计算总加工时间可以通过测试看效果
            A.append(sum([self.p[stage][j+1] for stage in range(1, 5 + 1)]))
        L = sorted(list(enumerate(A)), key=lambda x:x[1], reverse=True)
        for i in range(len(L)):
            L[i] = L[i][0] + 1
        sequence_1 = []
        sequence_1.append(L[0])
        for k in range(2, self.job_num + 1):
            min_Cmax = 1000000
            best_pos = -1
            # 第一次插入，为工件L[k-1]找到最佳的插入位置
            for insert_pos in range(k):
                sequence_1.insert(insert_pos, L[k-1])
                Cmax = self.cal_Cmax(sequence_1)
                sequence_1.remove(L[k-1])
                if Cmax < min_Cmax:
                    min_Cmax = Cmax
                    best_pos = insert_pos
            sequence_1.insert(best_pos, L[k-1])
            # 对插入位置前后（如存在）的工件进行重新插入
            if best_pos - 1 < 0:
                reinsert_job = [sequence_1[best_pos+1]]
            elif best_pos + 1 == len(sequence_1):
                reinsert_job = [sequence_1[best_pos-1]]
            else:
                reinsert_job = [sequence_1[best_pos-1], sequence_1[best_pos+1]]
            for job in reinsert_job:
                sequence_1.remove(job)
            # 对reinsert_job中的job进行重新插入，要求所有可能的位置都被覆盖到
            min_Cmax = 1000000
            best_pos = -1
            if len(reinsert_job) == 1:
                job = reinsert_job[0]
                for insert_pos in range(k):
                    sequence_1.insert(insert_pos, job)
                    Cmax = self.cal_Cmax(sequence_1)
                    sequence_1.remove(job)
                    if Cmax < min_Cmax:
                        min_Cmax = Cmax
                        best_pos = insert_pos
                sequence_1.insert(best_pos, job)
            else:
                available_pos = [i for i in range(k)]
                choices = list(itertools.permutations(available_pos, 2))
                for choice in choices:
                    if choice[0] < choice[1]:
                        sequence_1.insert(choice[0], reinsert_job[0])
                        sequence_1.insert(choice[1], reinsert_job[1])
                    else:
                        sequence_1.insert(choice[1], reinsert_job[1])
                        sequence_1.insert(choice[0], reinsert_job[0])
                    Cmax = self.cal_Cmax(sequence_1)
                    sequence_1.remove(reinsert_job[0])
                    sequence_1.remove(reinsert_job[1])
                    if Cmax < min_Cmax:
                        min_Cmax = Cmax
                        best_pos = choice
                if best_pos[0] < best_pos[1]:
                    sequence_1.insert(best_pos[0], reinsert_job[0])
                    sequence_1.insert(best_pos[1], reinsert_job[1])
                else:
                    sequence_1.insert(best_pos[1], reinsert_job[1])
                    sequence_1.insert(best_pos[0], reinsert_job[0])

        sequence_1.insert(0, 0)  # 调整工件编号使之与索引一致

        C = [[0 for i in range(len(sequence_1))] for j in
             range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间

        for order in range(1, len(sequence_1)):
            for stage in range(1, 5 + 1):
                front = sequence_1[order - 1]
                job = sequence_1[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + \
                                self.s[front][job] + self.p[stage][job]

        # 再确定工位6上的机器选择
        machine_choices = [0 for i in range(len(sequence_1))]

        front_in_stage_6 = [0, 0]
        for order in range(1, len(sequence_1)):
            job = sequence_1[order]
            if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and C[5][
                job] < C[6][front_in_stage_6[0]]:
                # 此时只能选工位6中的第二台机器
                machine_choices[job] = 1
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + \
                            self.s[front_in_stage_6[1]][job] + 1.2 * self.p[6][
                                job]
                front_in_stage_6[1] = job
            elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and C[5][
                job] < C[6][front_in_stage_6[1]]:
                # 此时只能选工位6中的第一台机器
                machine_choices[job] = 0
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + \
                            self.s[front_in_stage_6[0]][job] + self.p[6][job]
                front_in_stage_6[0] = job
            else:
                t0 = self.s[front_in_stage_6[0]][job] + self.p[6][job]
                t1 = self.s[front_in_stage_6[1]][job] + self.p[6][job] * 1.2
                m = 0 if t0 < t1 else 1
                machine_choices[job] = m
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + \
                            self.s[front_in_stage_6[m]][job] + \
                            (1 + m * 0.2) * self.p[6][job]
                front_in_stage_6[m] = job
        complete_time_in_stage_6 = C[6]
        # 最后确定工位7-10上加工顺序
        sequence_2 = [c[0] for c in sorted(enumerate(complete_time_in_stage_6),
                                           key=lambda x: x[1])]
        return sequence_1[1:], sequence_2[1:], machine_choices[1:]


class IOBS:
    """
        FRB4_1算法的改进版本，进行重新插入的不是新插入的工件在加工序列前后的工件，而是在插入
    该工件之前进行插入的2个工件（如存在）。

    参考文献：Constraint based local search for flowshops with sequence-dependent
    setup times.
    """
    def __init__(self, pt, st):
        self.p = pt
        self.s = st
        self.job_num = len(st) - 1

    def cal_Cmax(self, seq):
        seq = [0] + seq
        C = [[0 for i in range(self.job_num + 1)] for j in range(5 + 1)]
        Cmax = 0
        for order in range(1, len(seq)):
            for stage in range(1, 5 + 1):
                front = seq[order - 1]
                job = seq[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + self.s[front][job] + self.p[stage][job]
                Cmax = C[stage][job]
        return Cmax

    def solve(self):
        A = []
        for j in range(self.job_num):
            # 用1-5工位还是1-10工位上的加工时间计算总加工时间可以通过测试看效果
            A.append(sum([self.p[stage][j+1] for stage in range(1, 5 + 1)]))
        L = sorted(list(enumerate(A)), key=lambda x:x[1], reverse=True)
        for i in range(len(L)):
            L[i] = L[i][0] + 1
        sequence_1 = []
        sequence_1.append(L[0])
        for k in range(2, self.job_num + 1):
            min_Cmax = 1000000
            best_pos = -1
            # 第一次插入，为工件L[k-1]找到最佳的插入位置
            for insert_pos in range(k):
                sequence_1.insert(insert_pos, L[k-1])
                Cmax = self.cal_Cmax(sequence_1)
                sequence_1.remove(L[k-1])
                if Cmax < min_Cmax:
                    min_Cmax = Cmax
                    best_pos = insert_pos
            sequence_1.insert(best_pos, L[k-1])
            # 对插入位置前后（如存在）的工件进行重新插入
            if k == 2:
                reinsert_job = [L[k-2]]
            else:
                reinsert_job = [L[k-2], L[k-3]]
            for job in reinsert_job:
                sequence_1.remove(job)
            # 对reinsert_job中的job进行重新插入，要求所有可能的位置都被覆盖到
            min_Cmax = 1000000
            best_pos = -1
            if len(reinsert_job) == 1:
                job = reinsert_job[0]
                for insert_pos in range(k):
                    sequence_1.insert(insert_pos, job)
                    Cmax = self.cal_Cmax(sequence_1)
                    sequence_1.remove(job)
                    if Cmax < min_Cmax:
                        min_Cmax = Cmax
                        best_pos = insert_pos
                sequence_1.insert(best_pos, job)
            else:
                available_pos = [i for i in range(k)]
                choices = list(itertools.permutations(available_pos, 2))
                for choice in choices:
                    if choice[0] < choice[1]:
                        sequence_1.insert(choice[0], reinsert_job[0])
                        sequence_1.insert(choice[1], reinsert_job[1])
                    else:
                        sequence_1.insert(choice[1], reinsert_job[1])
                        sequence_1.insert(choice[0], reinsert_job[0])
                    Cmax = self.cal_Cmax(sequence_1)
                    sequence_1.remove(reinsert_job[0])
                    sequence_1.remove(reinsert_job[1])
                    if Cmax < min_Cmax:
                        min_Cmax = Cmax
                        best_pos = choice
                if best_pos[0] < best_pos[1]:
                    sequence_1.insert(best_pos[0], reinsert_job[0])
                    sequence_1.insert(best_pos[1], reinsert_job[1])
                else:
                    sequence_1.insert(best_pos[1], reinsert_job[1])
                    sequence_1.insert(best_pos[0], reinsert_job[0])

        sequence_1.insert(0, 0)  # 调整工件编号使之与索引一致

        C = [[0 for i in range(len(sequence_1))] for j in
             range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间

        for order in range(1, len(sequence_1)):
            for stage in range(1, 5 + 1):
                front = sequence_1[order - 1]
                job = sequence_1[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + \
                                self.s[front][job] + self.p[stage][job]

        # 再确定工位6上的机器选择
        machine_choices = [0 for i in range(len(sequence_1))]

        front_in_stage_6 = [0, 0]
        for order in range(1, len(sequence_1)):
            job = sequence_1[order]
            if C[6][front_in_stage_6[0]] > C[6][front_in_stage_6[1]] and \
                    C[5][
                        job] < C[6][front_in_stage_6[0]]:
                # 此时只能选工位6中的第二台机器
                machine_choices[job] = 1
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[1]]) + \
                            self.s[front_in_stage_6[1]][job] + 1.2 * \
                            self.p[6][
                                job]
                front_in_stage_6[1] = job
            elif C[6][front_in_stage_6[0]] < C[6][front_in_stage_6[1]] and \
                    C[5][
                        job] < C[6][front_in_stage_6[1]]:
                # 此时只能选工位6中的第一台机器
                machine_choices[job] = 0
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[0]]) + \
                            self.s[front_in_stage_6[0]][job] + self.p[6][
                                job]
                front_in_stage_6[0] = job
            else:
                t0 = self.s[front_in_stage_6[0]][job] + self.p[6][job]
                t1 = self.s[front_in_stage_6[1]][job] + self.p[6][job] * 1.2
                m = 0 if t0 < t1 else 1
                machine_choices[job] = m
                C[6][job] = max(C[5][job], C[6][front_in_stage_6[m]]) + \
                            self.s[front_in_stage_6[m]][job] + \
                            (1 + m * 0.2) * self.p[6][job]
                front_in_stage_6[m] = job
        complete_time_in_stage_6 = C[6]
        # 最后确定工位7-10上加工顺序
        sequence_2 = [c[0] for c in
                      sorted(enumerate(complete_time_in_stage_6),
                             key=lambda x: x[1])]
        return sequence_1[1:], sequence_2[1:], machine_choices[1:]


if __name__ == "__main__":
    import time
    from helper.read_data import get_info_from_file
    from helper.utils import test_feasiblity_and_cal_Cmax
    p, s = get_info_from_file('data/case 2.xlsx', print_table=True)

    start_time = time.time()
    sq1, sq2, m_choices = SPTCH(p, s).solve()
    obj = test_feasiblity_and_cal_Cmax(p, s, sq1, sq2, m_choices)
    print("SPTCH: ", obj)
    print(sq1, sq2, m_choices)

    sq1, sq2, m_choices = FTMIH(p, s).solve()
    obj = test_feasiblity_and_cal_Cmax(p, s, sq1, sq2, m_choices)
    print("FTMIH: ", obj)
    print(sq1, sq2, m_choices)

    sq1, sq2, m_choices = JohnsonsRule(p, s).solve()
    obj = test_feasiblity_and_cal_Cmax(p, s, sq1, sq2, m_choices)
    print("JohnsonRule: ", obj)
    print(sq1, sq2, m_choices)

    sq1, sq2, m_choices = FRB4_1(p, s).solve()
    obj = test_feasiblity_and_cal_Cmax(p, s, sq1, sq2, m_choices)
    print("FRB4_1: ", obj)
    print(sq1, sq2, m_choices)

    sq1, sq2, m_choices = IOBS(p, s).solve()
    obj = test_feasiblity_and_cal_Cmax(p, s, sq1, sq2, m_choices)
    print("IOBS: ", obj)
    print(sq1, sq2, m_choices)

    print(time.time() - start_time)
