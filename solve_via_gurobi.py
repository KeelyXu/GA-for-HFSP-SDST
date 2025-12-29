# Gurobi建模与求解
import gurobipy
from gurobipy import max_, quicksum

from helper.read_data import get_info_from_file

# 要修改求解的case，只需修改下面这行传入的文件名即可
p, s = get_info_from_file('data/case 1.xlsx', print_table=True)
n = len(s) - 1

model = gurobipy.Model('制造系统问题求解')

x = model.addVars(n+2, n+2, vtype=gurobipy.GRB.BINARY)
y = model.addVars(2, n+2, n+2, vtype=gurobipy.GRB.BINARY)
z = model.addVars(n+2, n+2, vtype=gurobipy.GRB.BINARY)
B = model.addVars(n+1, 11, vtype=gurobipy.GRB.CONTINUOUS, lb=0.0)
C = model.addVars(n+1, 11, vtype=gurobipy.GRB.CONTINUOUS, lb=0.0)
w = model.addVars(n, vtype=gurobipy.GRB.BINARY)     # 用于if then语句的辅助变量
Cmax = model.addVar(lb=0.0, ub=gurobipy.GRB.INFINITY,
                    vtype=gurobipy.GRB.CONTINUOUS, name="", column=None, obj=1.0)
# gurobi的max_函数要求所有参数为单个变量，因此引入辅助变量
Last_C = model.addVars(n, 10, vtype=gurobipy.GRB.CONTINUOUS, lb=0.0)

M = 10e6

# model.update()

model.setObjective(Cmax, gurobipy.GRB.MINIMIZE)

model.addConstrs(C[i, 0] == 0 for i in range(n+1))
model.addConstrs(C[0, k] == 0 for k in range(11))
model.addConstrs(B[0, k] == 0 for k in range(11))

model.addConstrs(quicksum(x[i, j] for i in range(n+1)) == 1 for j in range(1, n+2))
model.addConstrs(quicksum(x[i, j] for j in range(1, n+2)) == 1 for i in range(n+1))
model.addConstrs(x[i, i] == 0 for i in range(n+2))
model.addConstrs(x[i, 0] == 0 for i in range(n+2))
model.addConstrs(x[n+1, i] == 0 for i in range(n+2))

# 每个实际要被加工的工件只能选择一台机器（在工位6两台机器上有且仅有1个直接前驱，1个直接后继）
# 注意：经过测试，仅用下面两个约束描述机器的选择是不够充分的
# 例如，会出现以下情况：
# 机器0上：工件0的后面是工件2，工件1的后面是工件4
# 机器1上：工件0的后面是工件3，工件2的后面是工件1，工件3的后面是工件4
model.addConstrs(quicksum((y[0, i, j] + y[1, i, j]) for i in range(0, n+1)) == 1 for j in range(1, n+1))
model.addConstrs(quicksum((y[0, i, j] + y[1, i, j]) for j in range(1, n+2)) == 1 for i in range(1, n+1))
# 应添加约束：对工件1-n，如果有前驱，就一定要有后继
model.addConstrs(quicksum(y[0, i, job] for i in range(0, n+1)) == quicksum(y[0, job, j] for j in range(1, n+2)) for job in range(1, n+1))
model.addConstrs(quicksum(y[1, i, job] for i in range(0, n+1)) == quicksum(y[1, job, j] for j in range(1, n+2)) for job in range(1, n+1))

model.addConstr(quicksum(y[0, 0, j] for j in range(n+2)) == 1, name="")
model.addConstr(quicksum(y[1, 0, j] for j in range(n+2)) == 1, name="")
model.addConstr(quicksum(y[0, i, n+1] for i in range(n+2)) == 1, name="")
model.addConstr(quicksum(y[1, i, n+1] for i in range(n+2)) == 1, name="")
model.addConstrs(y[0, i, 0] == 0 for i in range(n+2))
model.addConstrs(y[1, i, 0] == 0 for i in range(n+2))
model.addConstrs(y[0, n+1, j] == 0 for j in range(n+2))
model.addConstrs(y[1, n+1, j] == 0 for j in range(n+2))
model.addConstrs(y[0, i, i] == 0 for i in range(n+2))
model.addConstrs(y[1, i, i] == 0 for i in range(n+2))

model.addConstrs(quicksum(z[i, j] for i in range(n+1)) == 1 for j in range(1, n+2))
model.addConstrs(quicksum(z[i, j] for j in range(1, n+2)) == 1 for i in range(n+1))
model.addConstrs(z[i, i] == 0 for i in range(n+2))
model.addConstrs(z[i, 0] == 0 for i in range(n+2))
model.addConstrs(z[n+1, i] == 0 for i in range(n+2))

for j in range(1, n+1):
    # 工位1-5
    for k in range(1, 6):
        model.addConstr(Last_C[j-1, k-1] >= quicksum(x[i, j] * C[i, k] for i in range(n+1)), name="")
        model.addConstr(B[j, k] == max_(C[j, k-1], Last_C[j-1, k-1]), name="")
        model.addConstr(C[j, k] >= B[j, k] + quicksum(x[i, j] * s[i][j] for i in range(n+1)) + p[k][j], name="")

    # 工位6
    model.addConstr(Last_C[j-1, 5] >= quicksum((y[0, i, j] + y[1, i, j]) * C[i, 6] for i in range(n+1)), name="")
    model.addConstr(B[j, 6] == max_(C[j, 5], Last_C[j-1, 5]), name="")
    # 下面的式子作了更新，文档里可能要修改
    model.addConstr(C[j, 6] >= B[j, 6] + quicksum((y[0, i, j] + y[1, i, j]) * s[i][j] for i in range(n+1)) +
                    (quicksum(y[0, i, j] for i in range(n+1)) + 1.2 * quicksum(y[1, i, j] for i in range(n+1))) * p[6][j], name="")

    # 工位7
    model.addConstr(Last_C[j-1, 6] >= quicksum(z[i, j] * C[i, 7] for i in range(n+1)), name="")
    model.addConstr(B[j, 7] == max_(C[j, 6], Last_C[j-1, 6]), name="")
    model.addConstr(C[j, 7] >= B[j, 7] + quicksum(z[i, j] * s[i][j] for i in range(n+1)) + p[7][j], name="")

    # 工位8-10
    for k in range(8, 11):
        model.addConstr(Last_C[j-1, k-1] >= quicksum(z[i, j] * C[i, k] for i in range(n+1)), name="")
        model.addConstr(B[j, k] == max_(C[j, k-1], Last_C[j-1, k-1]), name="")
        model.addConstr(C[j, k] >= B[j, k] + quicksum(z[i, j] * s[i][j] for i in range(n+1)) + p[k][j], name="")

for j in range(1, n+1):
    model.addConstr(B[j, 6] >= quicksum(x[i, j] * B[i, 6] for i in range(n+1)), name="")

for j in range(1, n+1):
    model.addConstr(quicksum(x[i, j] * C[i, 6] for i in range(n+1)) - C[j, 5] + 0.0001 <= M * (1 - w[j-1]), name="")
    model.addConstr(quicksum((y[0, i, j] + y[1, i, j]) * C[i, 6] for i in range(n+1)) -
                     quicksum(x[i, j] * C[i, 6] for i in range(n+1)) <= M * w[j-1], name="")

model.addConstrs(Cmax >= C[i, 10] for i in range(1, n+1))

model.setParam('IntegralityFocus', 1)
model.setParam('TimeLimit', 1800)
model.setParam('MIPFocus', 1)   # 对case 2要求先快速寻找可行解
model.optimize()
# model.computeIIS()
# model.write("model.ilp")

# 查看单目标规划模型的目标函数值
print("\nOptimal Objective Value", model.objVal)
# 查看变量取值
print('\nx values:')
for i in range(n+2):
    for j in range(n+2):
        if round(x[i, j].x) == 1:
            print('工件{}的后面是工件{}'.format(i, j))

print('\ny0 values:')
for i in range(n+2):
    for j in range(n + 2):
        if round(y[0, i, j].x) == 1:
            print('工件{}的后面是工件{}'.format(i, j))

print('\ny1 values:')
for i in range(n+2):
    for j in range(n + 2):
        if round(y[1, i, j].x) == 1:
            print('工件{}的后面是工件{}'.format(i, j))

print('\nz values:')
for i in range(n+2):
    for j in range(n + 2):
        if round(z[i, j].x) == 1:
            print('工件{}的后面是工件{}'.format(i, j))

print('\nC values:')
for i in range(n+1):
    print([C[i, j].x for j in range(11)])
