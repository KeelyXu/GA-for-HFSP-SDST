# 遗传算法实现
import random
from itertools import combinations, permutations
from helper.utils import test_feasiblity_and_cal_Cmax, get_best_machine_choices, list_all_feasible_machine_choices
from initial_solutions import FTMIH, FRB4_1, IOBS


class Genetic_Algorithm:
    """
    编码方式：工位1-5上加工顺序与工位7-10上加工顺序连接成的序列。
    注意：未将机器选择情况进行编码，因为对机器选择情况进行“交叉”是没有意义的。机器选择情况
    将由对应的解码算法生成。
    """

    def __init__(self, pt, st, struct):
        """
        struct参数为dict类型，key及value的取值如下。
        crossOperator: "OX"或"SB2OX"或"SL20X";
        localSearch: [1, 2, 3]及其任意非空子集;
        restart_method: "restart_bad"或"restart_good_and_bad"或"restart_with_diversity_control";
        accept_rule: 1或者2。
        """
        self.p = pt
        self.s = st
        self.job_num = len(st) - 1
        self.chromosome_pool = []     # 每个chromosome以(目标函数值，编码)的tuple形式存储（以后以chromosome和chromosome_code对二者进行区分）
        self.best_obj = 1000000
        self.best_chromosome = None
        self.decoder = get_best_machine_choices

        # 交叉算子的选择
        assert struct["crossOperator"] == "OX" or struct["crossOperator"] == "SB2OX" or struct["crossOperator"] == "SL2OX"
        if struct["crossOperator"] == "OX":
            self.cross_operator = self.OX
        elif struct["crossOperator"] == "SB2OX":
            self.cross_operator = self.SB2OX
        else:
            self.cross_operator = self.SL2OX
        # 变异时可用的邻域结构，1表示插入算法，2表示交换，3表示滑窗shuffle
        assert struct["localSearch"] == [1] or struct["localSearch"] == [2] or struct["localSearch"] == [3] \
               or struct["localSearch"] == [1, 2] or struct["localSearch"] == [1, 3] or \
               struct["localSearch"] == [2, 3] or struct["localSearch"] == [1, 2, 3]
        self.neighbour_struct = struct["localSearch"]
        # 最优解长时间没有改进后的种群更新策略
        assert struct["restart_method"] == "restart_bad" or struct["restart_method"] == "restart_good_and_bad" \
               or struct["restart_method"] == "restart_with_diversity_control"
        if struct["restart_method"] == "restart_bad":
            self.update_pool = self.restart_bad_solutions
        elif struct["restart_method"] == "restart_good_and_bad":
            self.update_pool = self.restart_good_and_bad_solutions
        else:
            self.update_pool = self.restart_with_diversity_control
        # 种群多样性控制策略，不允许插入过于相似的解
        assert struct["accept_rule"] == 1 or struct["accept_rule"] == 2
        if struct["accept_rule"] == 1:
            self.accept_rule = self.accept_rule_1
        else:
            self.accept_rule = self.accept_rule_2

    def get_obj_for_chromosome(self, chromosome_code):
        """
        根据染色体编码情况，利用解码器解码出机器选择序列，计算对应目标函数值。

        :param chromosome_code: 染色体编码。
        :return: 目标函数值。
        """
        sq1 = chromosome_code[:self.job_num]
        sq2 = chromosome_code[self.job_num:]
        m_choices = self.decoder(self.p, self.s, sq1, sq2)
        obj = test_feasiblity_and_cal_Cmax(self.p, self.s, sq1, sq2, m_choices)
        return obj

    def generate_chromosome_via_heuristic(self, heuristic):
        """
        利用构造型启发式快速构建较高质量的解。

        :param heuristic: 采用的启发式算法。
        :return: 带目标函数值的个体。
        """
        sq1, sq2, m_choices = heuristic(self.p, self.s).solve()
        chromosome_code = sq1 + sq2
        obj = self.get_obj_for_chromosome(chromosome_code)
        return obj, chromosome_code

    def generate_chromosome_randomly(self):
        """
        随机生成个体。
        注意：由于题目设置，通常较优解的seq1和seq2是比较接近的，所以在随机生成初始群落时，
        为加快搜索速度，个体的seq1与seq2相同。

        :return: 随机生成的带目标函数值的个体。
        """
        seq = [j + 1 for j in range(self.job_num)]
        random.shuffle(seq)
        chromosome_code = seq + seq
        obj = self.get_obj_for_chromosome(chromosome_code)
        return obj, chromosome_code

    def OX(self, P1, P2):
        length = self.job_num
        cuttings = random.sample([i for i in range(self.job_num)], 2)
        cuttings.sort()
        cut1 = cuttings[0]
        cut2 = cuttings[1]
        C1 = []
        C2 = []
        for p1, p2 in [(P1[:self.job_num], P2[:self.job_num]), (P1[self.job_num:], P2[self.job_num:])]:
            # 生成交叉点位
            c1 = [-1 for i in range(length)]
            c2 = [-1 for i in range(length)]
            c1[cut1:cut2 + 1] = p1[cut1:cut2 + 1]
            c2[cut1:cut2 + 1] = p2[cut1:cut2 + 1]
            i = cut2 + 1
            i = i % length
            j = i
            while i != cut1:
                while p2[j] in c1:
                    j = (j + 1) % length
                c1[i] = p2[j]
                i = (i + 1) % length
                j = (j + 1) % length

            i = cut2 + 1
            i = i % length
            j = i
            while i != cut1:
                while p1[j] in c2:
                    j = (j + 1) % length
                c2[i] = p1[j]
                i = (i + 1) % length
                j = (j + 1) % length
            C1.extend(c1)
            C2.extend(c2)
        return C1, C2

    def SB2OX(self, P1, P2):
        """
        论文提出的一种更适合SDST Flowshop的交叉算子。
        参考文献：Solving the flowshop scheduling problem with sequence dependent
        setup times using advanced metaheuristics.
        """
        cuttings = random.sample([i for i in range(self.job_num)], 2)
        cuttings.sort()
        cut1 = cuttings[0]
        cut2 = cuttings[1]
        C1 = []
        C2 = []
        for p1, p2 in [(P1[:self.job_num], P2[:self.job_num]), (P1[self.job_num:], P2[self.job_num:])]:
            start = None
            c1 = [-1 for i in range(self.job_num)]
            c2 = [-1 for i in range(self.job_num)]
            for i in range(self.job_num):   # 按similar block进行保留
                if start is None:
                    if p1[i] == p2[i]:
                        start = i
                else:
                    if p1[i] != p2[i]:
                        end = i
                        if end != start + 1:
                            c1[start:end] = p1[start:end]
                            c2[start:end] = p1[start:end]
                        start = None
            c1[cut1:cut2 + 1] = p1[cut1:cut2 + 1]
            c2[cut1:cut2 + 1] = p2[cut1:cut2 + 1]
            j = 0
            for i in range(self.job_num):
                if c1[i] != -1:
                    continue
                while p2[j] in c1:
                    j = (j + 1) % self.job_num
                c1[i] = p2[j]

            j = 0
            for i in range(self.job_num):
                if c2[i] != -1:
                    continue
                while p1[j] in c2:
                    j = (j + 1) % self.job_num
                c2[i] = p1[j]
            C1.extend(c1)
            C2.extend(c2)
        return C1, C2

    def SL2OX(self, P1, P2):
        """
        根据对一些当前获得的高质量解的结构分析，发现高质量解之间通常有许多相似片段。只是片段
        间的排列顺序不同。
        对SB2OX进行改进，当两个父代之间的极大相同片段的个数以及各个余下的“零散”工件的总数
        不超过5个（此时全排列总数可控）时，进行全排列组合，选取最优的排列。
        """
        p1 = P1[:self.job_num]
        p2 = P2[:self.job_num]
        links_in_common = []
        for i in range(self.job_num - 1):
            job1 = p1[i]
            job2 = p1[i + 1]
            job1_in_seq_b = p2.index(job1)
            if job1_in_seq_b != self.job_num - 1 and p2[job1_in_seq_b + 1] == job2:
                links_in_common.append([job1, job2])
        segments_in_common = []
        if len(links_in_common) > 0:
            segment = links_in_common[0][:]
            i = 0
            while i < len(links_in_common) - 1:
                if segment[-1] == links_in_common[i + 1][0]:  # 可首尾相连
                    segment.append(links_in_common[i + 1][1])
                else:
                    segments_in_common.append(segment)
                    segment = links_in_common[i + 1][:]
                i += 1
        job_in_common = []
        for segment in segments_in_common:
            job_in_common.extend(segment)
        segments = segments_in_common
        alone_job_num = 0
        for job in range(1, self.job_num + 1):
            if job not in job_in_common:
                segments.append([job])
                alone_job_num += 1
        if alone_job_num > self.job_num * 0.25:
            C1, C2 = self.SB2OX(P1, P2)
        else:
            random.shuffle(segments)
            seq = []
            for segment in segments:
                seq.extend(segment)
            C1 = seq + seq
            random.shuffle(segments)
            seq = []
            for segment in segments:
                seq.extend(segment)
            C2 = seq + seq
        return C1, C2

    def accept_rule_1(self, new_chromosome, pool=None):
        """接收规则1：只要当前种群中没有相同个体，就接收该个体。"""
        if pool is None:
            pool = self.chromosome_pool
        if new_chromosome not in pool:
            return True
        else:
            return False

    def accept_rule_2(self, new_chromosome, pool=None):
        """接收规则2：当没有现有个体与新插入个体的目标函数值相差10以内时，接收该个体。"""
        if pool is None:
            pool = self.chromosome_pool
        for chromosome in pool:
            if abs(chromosome[0] - new_chromosome[0]) < 10:
                return False
        return True

    def calculate_completion_matrix(self, C, seq1, m_choices, cal_stage1to5=True):
        """
        根据传入工位1-5上的加工顺序及机器选择序列计算完工矩阵中可计算的部分。

        :param C: 完工时间矩阵。
        :param seq1: 工位1-5上加工顺序。
        :param m_choices: 工位6上机器选择序列。
        :param cal_stage1to5: 工位1-5上的完工时间是否需要计算。
        """
        seq1 = [0] + seq1
        if cal_stage1to5 is True:
            for order in range(1, self.job_num + 1):
                for stage in range(1, 5 + 1):
                    front = seq1[order - 1]
                    job = seq1[order]
                    C[stage][job] = max(C[stage - 1][job], C[stage][front]) + s[front][job] + p[stage][job]
        front_in_stage_6 = [0, 0]
        for order in range(1, self.job_num + 1):
            job = seq1[order]
            m_choice = m_choices[job - 1]
            front = front_in_stage_6[m_choice]
            C[6][job] = max(C[5][job], C[6][front]) + s[front][job] + (1 + m_choice * 0.2) * p[6][job]
            front_in_stage_6[m_choice] = job

    def calculate_makespan(self, C, seq2, upperbound=1000000):
        """
        计算makespan。

        :param C: 完工时间矩阵。
        :param seq2: 工位7-10上加工顺序。
        :param upperbound: 计算makespan过程中的上界值，若完工时间矩阵计算过程中某个元素
            值超过upperbound，立刻返回None。
        :return: makespan或None。
        """
        Cmax = 0
        seq2 = [0] + seq2
        for order in range(1, self.job_num + 1):
            for stage in range(7, 10 + 1):
                front = seq2[order - 1]
                job = seq2[order]
                C[stage][job] = max(C[stage - 1][job], C[stage][front]) + s[front][job] + p[stage][job]
                Cmax = C[stage][job]
                if Cmax >= upperbound:
                    return None
        return Cmax

    def mutation(self, chromosome, neighbour_choice=None, speed_up_level=(1, 1)):
        """
        编码变异（局部搜索优化），允许指定邻域结构。

        :param chromosome: 要进行mutation的个体。
        :param neighbour_choice: 指定选择哪种邻域结构（比如1）。若为None，随机选择。
        :param speed_up_level: 一个二元数组，两个元素的取值都只能为0/1。
            第一个0表示考虑seq2变化对机器序列选择的影响，1表示不考虑seq2变化对机器序列选择的影响；
            第二个0表示结束所有搜索并取最优解，1表示一旦找到最优解立即退出。
        """

        seq1 = chromosome[1][:self.job_num]
        seq2 = chromosome[1][self.job_num:]
        C = [[0 for i in range(self.job_num + 1)] for j in
             range(10 + 1)]  # C[i][j]表示工位i上工件j的完工时间

        if speed_up_level[0] == 1:    # 不考虑seq2变化对机器序列选择的影响
            m_choices = self.decoder(self.p, self.s, seq1, seq2)
            self.calculate_completion_matrix(C, seq1, m_choices)
        else:       # 考虑seq2变化对机器序列选择的影响
            feasible_choices = list_all_feasible_machine_choices(self.p, self.s, seq1, C)

        if neighbour_choice is None:
            neighbour_choice = random.choice(self.neighbour_struct)

        # 邻域结构1：插入算法。按照随机顺序选择工件进行重新插入，经测试，改善概率接近100%。
        if neighbour_choice == 1:
            min_obj = chromosome[0]
            insert_seq = seq2[:]
            random.shuffle(insert_seq)
            for job in insert_seq:
                initial_seq = seq2[:]
                seq2.remove(job)
                best_pos = None
                for pos in range(self.job_num - 1):
                    seq2.insert(pos, job)
                    if speed_up_level[0] == 0:
                        best_m_choices = self.decoder(self.p, self.s, seq1, seq2, C,
                                                                  feasible_choices)
                        self.calculate_completion_matrix(C, seq1, best_m_choices, False)
                    makespan = self.calculate_makespan(C, seq2, min_obj)
                    if makespan is not None:  # 如用贪婪法，这里即退出并替换解
                        if speed_up_level[1] == 1:
                            chromosome[1][self.job_num:] = seq2
                            chromosome = (self.get_obj_for_chromosome(chromosome[1]), chromosome[1])
                            return chromosome
                        min_obj = makespan
                        best_pos = pos
                    seq2.remove(job)
                if best_pos is not None:
                    seq2.insert(best_pos, job)
                else:
                    seq2 = initial_seq
            chromosome[1][self.job_num:] = seq2
            chromosome = (self.get_obj_for_chromosome(chromosome[1]), chromosome[1])
            return chromosome

        # 邻域结构2：交换算法。将2个工件加工顺序交换，经测试，改善概率接近100%。
        if neighbour_choice == 2:
            min_obj = chromosome[0]
            swap_choices = list(combinations(seq2, 2))
            random.shuffle(swap_choices)
            for swap_pair in swap_choices:
                index1 = seq2.index(swap_pair[0])
                index2 = seq2.index(swap_pair[1])
                seq2[index1], seq2[index2] = seq2[index2], seq2[index1]
                if speed_up_level[0] == 0:
                    best_m_choices = self.decoder(self.p, self.s, seq1, seq2, C,
                                                              feasible_choices)
                    self.calculate_completion_matrix(C, seq1, best_m_choices, False)
                makespan = self.calculate_makespan(C, seq2, min_obj)
                if makespan is not None:  # 如用贪婪法，这里即退出并替换解
                    if speed_up_level[1] == 1:
                        chromosome[1][self.job_num:] = seq2
                        chromosome = (self.get_obj_for_chromosome(chromosome[1]), chromosome[1])
                        return chromosome
                    min_obj = makespan
                else:  # 解没有得到改善，将顺序换回来
                    seq2[index1], seq2[index2] = seq2[index2], seq2[index1]
            chromosome[1][self.job_num:] = seq2
            chromosome = (self.get_obj_for_chromosome(chromosome[1]), chromosome[1])
            return chromosome

        # 邻域结构3：滑窗shuffle法。设置窗口大小为3。即将一个宽度为3的窗口从前向后滑动，窗口内
        # 的3个零件顺序可自由调整，窗口外的工件顺序保持不变。
        if neighbour_choice == 3:
            min_obj = chromosome[0]
            for i in range(self.job_num - 2):
                initial_seq = seq2[:]
                jobs_to_shuffle = seq2[i:i+3]
                shuffle_choices = list(permutations(jobs_to_shuffle, 3))
                for choice in shuffle_choices:
                    seq2[i:i+3] = choice
                    if speed_up_level[0] == 0:
                        best_m_choices = self.decoder(self.p, self.s, seq1, seq2, C,
                                                                  feasible_choices)
                        self.calculate_completion_matrix(C, seq1, best_m_choices, False)
                    makespan = self.calculate_makespan(C, seq2, min_obj)
                    if makespan is not None:  # 如用贪婪法，这里即退出并替换解
                        if speed_up_level[1] == 1:
                            chromosome[1][self.job_num:] = seq2
                            chromosome = (self.get_obj_for_chromosome(chromosome[1]), chromosome[1])
                            return chromosome
                        min_obj = makespan
                        initial_seq = seq2[:]
                seq2 = initial_seq
            chromosome[1][self.job_num:] = seq2
            chromosome = (self.get_obj_for_chromosome(chromosome[1]), chromosome[1])
            return chromosome

    def insort(self, population_size, individual):
        """
        用于维持插入后pool的有序性的函数。

        :param population_size: 当前种群大小。
        :param individual: 要插入种群中的个体。
        """
        for i in range(population_size):
            if individual[0] < self.chromosome_pool[i][0]:
                self.chromosome_pool.insert(i, individual)
                return
        self.chromosome_pool.insert(population_size, individual)

    def get_parent_via_tournament(self, population_size):
        """根据文献，对流水车间调度问题，锦标赛效果优于轮盘赌。"""
        # 随机选择父代
        candidate_1 = random.randint(0, population_size - 1)
        candidate_2 = random.randint(0, population_size - 1)
        while candidate_2 == candidate_1:   # 父代不应相同
            candidate_2 = random.randint(0, population_size - 1)
        if candidate_1 < candidate_2:
            return self.chromosome_pool[candidate_1]
        else:
            return self.chromosome_pool[candidate_2]

    def restart_with_diversity_control(self):
        """
        对种群进行更新，依概率保留当前种群中的优秀基因片段。
        参考文献：Effective metaheuristics for scheduling a hybrid flowshop
        with sequence-dependent setup times.
        """
        new_pool = []
        # 对各类link在种群中出现的次数进行统计
        job_block_matrix = [[0 for i in range(self.job_num)] for j in range(self.job_num)]
        for chromosome in self.chromosome_pool:
            for i in range(self.job_num - 1):
                job1 = chromosome[1][i]
                job2 = chromosome[1][i+1]
                job_block_matrix[job1-1][job2-1] += 1
        min_count = min([min(line) for line in job_block_matrix])
        max_count = max([max(line) for line in job_block_matrix])
        scaler = max_count - min_count
        pool_size = len(self.chromosome_pool)
        for id in range(int(pool_size*0.5)):    # 前50%
            chromosome_code = self.chromosome_pool[id][1]
            remain_probs = [0 for i in range(self.job_num)]
            for j in range(self.job_num):
                if j == 0:
                    numerator = job_block_matrix[chromosome_code[j]-1][chromosome_code[j+1]-1] - min_count
                elif j == self.job_num - 1:
                    numerator = job_block_matrix[chromosome_code[j-1]-1][chromosome_code[j]-1] - min_count
                else:
                    numerator = (job_block_matrix[chromosome_code[j-1]-1][chromosome_code[j]-1] + job_block_matrix[chromosome_code[j]-1][chromosome_code[j+1]-1]) / 2 - min_count
                remain_probs[j] = 0.1 + numerator / scaler * 0.8
            while True:
                new_chromosome_code = [-1 for i in range(self.job_num)]
                for i in range(self.job_num):
                    if random.random() < remain_probs[i]:
                        new_chromosome_code[i] = chromosome_code[i]
                missing_jobs = [i + 1 for i in range(self.job_num)]
                for job in new_chromosome_code:
                    if job in missing_jobs:
                        missing_jobs.remove(job)
                for i in range(self.job_num):
                    if new_chromosome_code[i] == -1:
                        random_job = random.choice(missing_jobs)
                        new_chromosome_code[i] = random_job
                        missing_jobs.remove(random_job)
                new_chromosome_code = new_chromosome_code + new_chromosome_code
                new_chromosome = (self.get_obj_for_chromosome(new_chromosome_code), new_chromosome_code)
                new_chromosome = self.mutation(new_chromosome)
                if self.accept_rule(new_chromosome, new_pool):
                    new_pool.append(new_chromosome)
                    break
        for id in range(int(pool_size*0.5), pool_size):     # 后50%
            while True:
                new_chromosome = self.generate_chromosome_randomly()
                new_chromosome = self.mutation(new_chromosome)
                if self.accept_rule(new_chromosome, new_pool):
                    new_pool.append(new_chromosome)
                    break
        new_pool.sort(key=lambda x:x[0])
        self.chromosome_pool = new_pool
        
    def restart_bad_solutions(self, percent=0.5):
        """
        刷新当前种群中的较差个体。

        :param percent: 刷新个体总数所占百分比。
        """
        np = len(self.chromosome_pool)
        # 更新方法：更新后50%
        self.chromosome_pool = self.chromosome_pool[:int(np * percent)]
        population_size = len(self.chromosome_pool)
        while population_size != np:
            chromosome = self.generate_chromosome_randomly()
            while chromosome in self.chromosome_pool:
                chromosome = self.generate_chromosome_randomly()
                # chromosome = self.mutation(chromosome, speed_up_level=(1, 1))
            self.chromosome_pool.append(chromosome)
            population_size += 1
        self.chromosome_pool.sort(key=lambda x:x[0])
        
    def restart_good_and_bad_solutions(self):
        """
        种群中前10%中更新5%（但保留最优解），并更新最差的30%。
        """
        np = len(self.chromosome_pool)
        if self.best_chromosome[0] > self.chromosome_pool[0][0]:
            self.best_chromosome = self.chromosome_pool[0][:]
        percent_0to10 = self.chromosome_pool[1:int(np * 0.1)]
        random.shuffle(percent_0to10)
        percent_10to70 = self.chromosome_pool[int(np * 0.1):int(np * 0.7)]
        self.chromosome_pool = [self.chromosome_pool[0]] + percent_0to10[:int(np * 0.05)] + percent_10to70
        population_size = len(self.chromosome_pool)
        while population_size != np:
            chromosome = self.generate_chromosome_randomly()
            while chromosome in self.chromosome_pool:
                chromosome = self.generate_chromosome_randomly()
                # chromosome = self.mutation(chromosome, speed_up_level=(1, 1))
            self.chromosome_pool.append(chromosome)
            population_size += 1
        self.chromosome_pool.sort(key=lambda x: x[0])

    def solve(self, np, ng, pm, gr):
        """
        问题求解。

        :param np: 种群规模。
        :param ng: 进化迭代次数。
        :param pm: 变异概率。
        :param gr: 容忍最优解没有改善的迭代次数上界。
        :return: 最优个体。
        """
        # 用启发式生成初始种群中的较优秀个体，经测试FTMIN, FRB4_1, IOBS三种启发式对case 1, 2效果较好。
        # for heuristic in [SPTCH, FTMIH, JohnsonsRule, FRB4_1, IOBS]:
        for heuristic in [FTMIH, FRB4_1, IOBS]:
            chromosome = self.generate_chromosome_via_heuristic(heuristic)
            if self.accept_rule(chromosome):
                self.chromosome_pool.append(chromosome)
        # print(self.chromosome_pool)

        # 随机生成余下初始个体种群
        pool_size = len(self.chromosome_pool)
        for i in range(np - pool_size):
            chromosome = self.generate_chromosome_randomly()
            while chromosome in self.chromosome_pool:
                chromosome = self.generate_chromosome_randomly()
            self.chromosome_pool.append(chromosome)

        # 维持群落的有序性
        self.chromosome_pool.sort(key=lambda x:x[0])
        self.best_chromosome = self.chromosome_pool[0]

        improve_after_mutation = 0
        improve_count = 0

        # 开始进化迭代
        no_improve_count = 0
        for iter in range(1, ng + 1):
            if iter % 100 == 0:
                print("iter: ", iter, "\tbest solution: ", self.best_chromosome)
            # 用二元锦标赛方法选出父代
            while True:
                P1 = self.get_parent_via_tournament(np)
                P2 = self.get_parent_via_tournament(np)
                while P2 == P1:     # 保证父代是不同的，杂交才有意义
                    P2 = self.get_parent_via_tournament(np)
                C1, C2 = self.cross_operator(P1[1], P2[1])
                obj1 = self.get_obj_for_chromosome(C1)
                obj2 = self.get_obj_for_chromosome(C2)
                if obj1 < obj2:
                    child = (obj1, C1)
                else:
                    child = (obj2, C2)
                if self.accept_rule(child):
                    break
            # 子代按概率产生突变
            if random.random() < pm:
                child_after_mutation = self.mutation(child, speed_up_level=(1, 0))
                if self.accept_rule(child_after_mutation):
                    child = child_after_mutation
                    if child[0] < self.chromosome_pool[0][0]:
                        improve_after_mutation += 1
            # 选择更优秀的子代，取代种群中质量较差的后半部分中的某个个体
            replace_index = random.randint(int(np / 2), np - 1)
            del self.chromosome_pool[replace_index]
            self.insort(np - 1, child)
            # 查看新种群中最优解是否发生变化，如长时间不变，考虑restart
            if self.chromosome_pool[0][0] < self.best_obj:
                self.best_obj = self.chromosome_pool[0][0]
                improve_count += 1
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= gr:      # 进行restart
                # print("restart...")
                self.update_pool()
                no_improve_count = 0
            if self.best_chromosome[0] > self.chromosome_pool[0][0]:
                self.best_chromosome = self.chromosome_pool[0][:]
        # 打印找到的最优解的summary
        print(self.best_chromosome[0])
        seq1 = self.best_chromosome[1][:self.job_num]
        seq2 = self.best_chromosome[1][self.job_num:]
        m_choices = self.decoder(self.p, self.s, seq1, seq2)
        print(seq1)
        print(seq2)
        print(m_choices)
        # print("improvement count: ", improve_count)
        # print("improvement after mutation: ", improve_after_mutation)
        return self.best_chromosome


if __name__ == "__main__":
    from helper.read_data import get_info_from_file
    import time

    p, s = get_info_from_file('data/case 2.xlsx', print_table=True)
    start_time = time.time()
    random.seed(21)     # 该随机数在case 1和case 2下均能找到最优解（要复现case 1，只要将上方文件名中"case 2"改为"case 1"）
    Genetic_Algorithm(p, s, {"crossOperator": "OX", "localSearch": [1, 2, 3], "restart_method": "restart_bad", "accept_rule": 1}).solve(50, 3000, 0.2, 1000)
    end_time = time.time()
    print("time: ", end_time - start_time)
