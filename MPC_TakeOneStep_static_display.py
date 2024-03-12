import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


m = gp.Model("MPC") # 创建模型 Create a new model
m.setParam('OutputFlag', 0) # 设置 gurobi 输出模式为不输出

# 1. 设置变量
# 预测区间 horizon
N = 15
# 变量约束 Bounds
V = 200
umin = np.array([-V, -V])
umax = np.array([V, V])
umax = np.tile(umax, (N,1))
umin = np.tile(umin, (N,1))
x = m.addMVar(shape=(N+1,2), lb=-GRB.INFINITY, name='x')
u = m.addMVar(shape=(N,2), lb=umin, ub=umax, name='u')
Target = m.addMVar(shape=(N+1,2),lb=-GRB.INFINITY, name='Target')
V_T = m.addMVar(shape=(N,2),lb=-GRB.INFINITY, name='V_T')
sint = m.addMVar(N,lb=-GRB.INFINITY, name='sint')
cost = m.addMVar(N,lb=-GRB.INFINITY, name='cost')
t = m.addMVar(N+1, name='t')
tt = m.addMVar(N+1, name='tt')
# m.write('addMVar.lp')

# 2. 设置目标函数 
obj = sum(np.square(x[k, 0]-Target[k, 0])+np.square(x[k, 1]-Target[k, 1]) for k in range(N+1))
m.setObjective(obj, GRB.MINIMIZE)
# m.write('setObj.lp')

# 3. 设置约束条件
x0 = np.array([-100., -200.])
T0 = np.array([0., 0.])
V_T0 = np.array([0., 0.])
t0 = 0
start_value_1 = m.addConstr(x[0]==x0)
Target_value_1 = m.addConstr(Target[0]==T0)
t_value = m.addConstr(t[0] == t0)
T = 0.1
for k in range(N):
    m.addConstr(t[k+1] == t[k] + 2)
    m.addConstr(tt[k] == 2*np.pi/360*t[k])
    m.addGenConstrCos(tt[k], cost[k])
    m.addGenConstrSin(tt[k], sint[k])
    m.addConstr(V_T[k, 0] == 200*sint[k])
    m.addConstr(V_T[k, 1] == 200*cost[k])
    m.addConstr(x[k+1, :] == x[k, :] + T * u[k, :])
    m.addConstr(Target[k+1, :] == Target[k, :] + T * V_T[k, :])
# m.write('addConstr.lp')

# 4. 迭代
mpc_iter = 0    #目前迭代次数
iteration = 100 #总迭代次数
# UAV 和 Target 的实际轨迹及两者之间的距离、u 的实际输入
xx = np.zeros(iteration+1)
xy = np.zeros(iteration+1)
Tx = np.zeros(iteration+1)
Ty = np.zeros(iteration+1)
xx[0] = x0[0]
xy[0] = x0[1]
Tx[0] = T0[0]
Ty[0] = T0[1]
distance = np.zeros(iteration+1)
ux = np.zeros(iteration)
uy = np.zeros(iteration)
# 开始仿真、迭代更新
while mpc_iter < iteration:
    m.optimize()
    xx[mpc_iter+1] = x.x[1][0]
    xy[mpc_iter+1] = x.x[1][1]
    Tx[mpc_iter+1] = Target.x[1][0]
    Ty[mpc_iter+1] = Target.x[1][1]
    ux[mpc_iter] = u.x[0][0]
    uy[mpc_iter] = u.x[0][1]
    print(mpc_iter)
    distance[mpc_iter] = np.sqrt(np.square(x.x[0][0]-Target.x[0][0])+np.square(x.x[0][1]-Target.x[0][1]))
    print(distance[mpc_iter])
    # 将求解后的下一步作为下一次迭代的初始值
    mpc_iter+=1
    m.remove(t_value)
    m.remove(start_value_1)
    m.remove(Target_value_1)
    start_value_1 = m.addConstr(x[0]==np.array(x.x[1]))
    Target_value_1 = m.addConstr(Target[0]==np.array(Target.x[1]))
    t_value = m.addConstr(t[0]==t.x[1])
    m.update()

# 画图：Target和UAV实际的轨迹
plt.figure()
# plt.subplot(2, 1, 1)
plt.plot(Tx, Ty, color="darkblue", linewidth=1, linestyle='--', label='Target', marker='+')
plt.plot(xx, xy, color="deeppink", linewidth=1, linestyle='--', label='UAV', marker='+')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectories of UAV and target")
plt.grid()
plt.legend(loc=1)

# 画图：Target和UAV实际的距离
plt.figure()
# plt.subplot(2, 1, 2)
plt.plot(range(iteration+1), distance, color="goldenrod", linewidth=1, linestyle='--', label='distance', marker='+')
plt.xlabel("t")
plt.ylabel("distance")
plt.title("Distance between UAV and target")
plt.grid()
plt.legend(loc=1)

# plt.subplots_adjust(wspace = 0.5, hspace = 0.5)#调整子图间距
plt.show()
