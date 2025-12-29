# Gurobi Model and Genetic Algorithm for the Hybrid Flow Shop Scheduling Problem with Sequence-Dependent Setup Times (HFSP-SDST) 

![Date](https://img.shields.io/badge/Completion_Date-November_2023-blue.svg)

This is my course project for SJTU-IE4901 (2023 Autumn).

## Problem Description

HFSP-SDST is a complex combinatorial optimization problem (NP-hard) frequently encountered in modern manufacturing. It involves determining the optimal machine assignment and job sequence across multiple production stages to minimize the **Makespan ($C_{max}$)**.

### Objectives

The goal is to determine the optimal **allocation of machines** and **sequence of jobs** at each stage to optimize the total makespan.

### Example

The following diagram illustrates a representative **Hybrid Flow Shop** configuration which is also used as the instance in this project.

- **Hybrid Stage (Stage 6)**: This stage features two parallel machines (**6-1** and **6-2**), representing a typical capacity bottleneck or flexibility point where a job can be assigned to either machine.

<img src="assets/image-20251229212706253.png" alt="image-20251229212706253" style="zoom: 50%;" />

#### Constraints & Rules

- **Processing Flow**: Every workpiece must pass through stations 1 to 10 in strict numerical order (no skipping).

- **Sequence Invariance**:
  - The processing sequence determined at Station 1 must remain identical through Stations 2–5.
  - The sequence determined at Station 7 must remain identical through Stations 8–10.

- **Station 6 Logic**: 
  - Workpieces from Station 5 must enter an available machine in Station 6 immediately if one is free (no waiting).
  - Processing time for M6-2 is 1.2x the time required for M6-1.

- **Reordering**: The buffer after Station 6 allows for the re-sequencing of workpieces before they enter Station 7.

- **Setup Time (Changeover)**:
  - Switching between different types of workpieces requires a specific setup time.
  - No pre-setup: Setup can only begin after the workpiece has arrived at the machine.
  - The first workpiece processed on any machine requires zero setup time.



To minimize the total completion time, the following must be determined:

1. **Input Sequence**: The optimal processing order for Stations 1–5.
2. **Station 6 Allocation**: The distribution of workpieces between the two parallel machines in Station 6.
3. **Output Sequence**: The optimal re-sequenced processing order for Stations 7–10.

## Mathematical Model

### Sets and Indices

$I$: Set of workpieces, $I = \{0, 1, \dots, n, n+1\}$, where $n$ is the total number of actual workpieces.

$K$: Set of stations, $K = \{0, 1, \dots, 10\}$.

$K_1$: Subset of stations 1–5, $K_1 = \{1, \dots, 5\}$.

$K_2$: Subset of stations 8–10, $K_2 = \{8, 9, 10\}$.

$i, j$: Indices for workpieces.

$k$: Index for stations.

$m$: Index for machines at Station 6, $m \in \{1, 2\}$.

### Parameters

$P_{i,k}$: Processing time of workpiece $i$ at station $k$ (where $P_{i,6}$ is the time on machine $m$ at Station 6).

$S_{i,j}$: Setup time required to switch from workpiece $i$ to workpiece $j$ on any machine.

$M$: A sufficiently large constant.

### Decision Variables

$x_{i,j}$: Binary variable; 1 if workpiece $i$ is the immediate predecessor of workpiece $j$ at stations 1–5, 0 otherwise.

$y_{m,i,j}$: Binary variable; 1 if workpiece $i$ is the immediate predecessor of workpiece $j$ on machine $m$ ($m=1, 2$) at Station 6, 0 otherwise.

$z_{i,j}$: Binary variable; 1 if workpiece $i$ is the immediate predecessor of workpiece $j$ at stations 7–10, 0 otherwise.

$B_{i,6}$: Start time of workpiece $i$ at Station 6.

$C_{i,k}$: Completion time of workpiece $i$ at station $k$.

$w_{i}$: Auxiliary binary variable used to linearize conditional logic.

### Model

$$\min(obj) = \min C_{n+1,10}$$

**Initial Conditions**
$$
C_{i,0} = 0,\qquad \forall i \in I \tag{1}
$$

$$
C_{0,k} = 0,\qquad \forall k \in K \tag{2}
$$

$$
B_{0,k} = 0,\qquad \forall k \in K \tag{3}
$$

#### Sequence Constraints (Stations 1–5)

$$
\sum_{i=0}^{n}x_{i,j} = 1,\qquad \forall j \in I \setminus \{0\} \tag{4}
$$

$$
\sum_{j=1}^{n+1}x_{i,j} = 1, \qquad \forall i \in I \setminus \{n+1\} \tag{5}
$$

$$
x_{i,i} = 0, \qquad \forall i \in I \tag{6}
$$

$$
\sum_{i=0}^{n+1}x_{i,0} + \sum_{j=0}^{n+1}x_{n+1,j} = 0 \tag{7}
$$

#### Sequence Constraints (Station 6 Machines)

$$
\sum_{i=0}^{n}(y_{1,i,j} + y_{2,i,j}) = 1,\qquad \forall j \in I \setminus \{0, n+1\} \tag{8}
$$

$$
\sum_{j=1}^{n+1}(y_{1,i,j} + y_{2,i,j}) = 1,\qquad \forall i \in I \setminus \{0, n+1\} \tag{9}
$$

$$
y_{m,i,i} = 0,\qquad \forall i \in I, m \in \{1, 2\} \tag{10}
$$

$$
\sum_{i=0}^{n+1}y_{m,i,0} + \sum_{j=0}^{n+1}y_{m,n+1,j} = 0,\qquad \forall m \in \{1, 2\} \tag{11}
$$

$$
\sum_{i=0}^{n}y_{m,i,p} = \sum_{j=1}^{n+1}y_{m,p,j},\quad \forall p \in I \setminus \{0, n+1\}, m \in \{1, 2\} \tag{12}
$$

$$
\sum_{j=0}^{n+1}y_{m,0,j} = 1,\qquad \forall m \in \{1, 2\} \tag{13}
$$

$$
\sum_{i=0}^{n+1}y_{m,i,n+1} = 1,\qquad \forall m \in \{1, 2\} \tag{14}
$$

#### Sequence Constraints (Stations 7–10)

$$
\sum_{i=0}^{n}z_{i,j} = 1,\qquad \forall j \in I \setminus \{0\} \tag{15}
$$

$$
\sum_{j=1}^{n+1}z_{i,j} = 1,\qquad \forall i \in I \setminus \{n+1\} \tag{16}
$$

$$
z_{i,i} = 0,\qquad \forall i \in I \tag{17}
$$

$$
\sum_{i=0}^{n+1}z_{i,0} + \sum_{j=0}^{n+1}z_{n+1,j} = 0 \tag{18}
$$

#### Completion Time and Start Time Constraints

$$
C_{j,k} = \max(C_{j,k-1}, \sum_{i=0}^{n}x_{i,j} \cdot C_{i,k}) + \sum_{i=0}^{n}x_{i,j} \cdot S_{i,j} + P_{j,k},\qquad \forall j \in I \setminus \{0\}, k \in K_1 \tag{19}
$$

$$
C_{j,k} = \max(C_{j,k-1}, \sum_{i=0}^{n}z_{i,j} \cdot C_{i,k}) + \sum_{i=0}^{n}z_{i,j} \cdot S_{i,j} + P_{j,k},\qquad \forall j \in I \setminus \{0\}, k \in K_2 \tag{20}
$$

$$
C_{j,7} \ge \max(C_{j,6}, \sum_{i=0}^{n}z_{i,j} \cdot C_{i,7}) + \sum_{i=0}^{n}x_{i,j} \cdot S_{i,j} + P_{j,7},\qquad \forall j \in I \setminus \{0\} \tag{21}
$$

$$
C_{j,6} = B_{j,6} + \sum_{i=0}^{n}(y_{1,i,j} + y_{2,i,j}) \cdot S_{i,j} + \sum_{i=0}^{n}(y_{1,i,j} + y_{2,i,j} \cdot 1.2) \cdot P_{i,6},\qquad \forall j \in I \setminus \{0\} \tag{22}
$$

$$
B_{j,6} = \max(C_{j,5}, \sum_{i=0}^{n}(y_{1,i,j} + y_{2,i,j}) \cdot C_{i,6}),\qquad \forall j \in I \setminus \{0\} \tag{23}
$$

$$
B_{j,6} \ge \sum_{i=0}^{n}x_{i,j} B_{i,6},\qquad \forall j \in I \setminus \{0, n+1\} \tag{24}
$$

#### Linearized Logical Constraints

$$
\sum_{i=0}^{n}x_{i,j} C_{i,6} - C_{j,5} + 0.00001 \le M(1 - w_j),\qquad \forall j \in I \setminus \{0, n+1\} \tag{25}
$$

$$
\sum_{i=0}^{n}(y_{1,i,j} + y_{2,i,j}) \cdot C_{i,6} - \sum_{i=0}^{n}x_{i,j} C_{i,6} \le M \cdot w_j,\qquad \forall j \in I \setminus \{0, n+1\} \tag{26}
$$

## Algorithm Overview

The algorithm is based on the classic **Genetic Algorithm (GA)** framework with *some configurable options*. Five simple heuristics are implemented to construct initial solutions:

1. SPTCH
2. FTMIH
3. Johnson's Rule
4. FRB4<sub>1</sub>
5. IOBS

## Reference

1. Pan, Q. K., Gao, L., Li, X. Y., & Gao, K. Z. (2017). Effective metaheuristics for scheduling a hybrid flowshop with sequence-dependent setup times. *Applied Mathematics and Computation*, *303*, 89-112.
2. Ruiz, R., Maroto, C., & Alcaraz, J. (2005). Solving the flowshop scheduling problem with sequence dependent setup times using advanced metaheuristics. *European Journal of Operational Research*, *165*(1), 34-54.
3. Kurz, M. E., & Askin, R. G. (2004). Scheduling flexible flow lines with sequence-dependent setup times. *European Journal of Operational Research*, *159*(1), 66-82.
4. Riahi, V., Newton, M. H., & Sattar, A. (2021). Constraint based local search for flowshops with sequence-dependent setup times. *Engineering Applications of Artificial Intelligence*, *102*, 104264.