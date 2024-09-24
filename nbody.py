import casadi as ca
from casadi import MX, vertcat, integrator
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

n = 100
min_rsr = 0.1
G = 1
d = 3
q = MX.sym('q', 2*d*n)

tf = 50
nt = 500

vdot = []
for i in range(n):
    vdot.append(ca.DM.zeros(d))
    for j in range(n):
        if i==j: continue
        dx = q[(d*i):(d*i+d)]-q[(d*j):(d*j+d)]
        vdot[-1] += -dx/(ca.fmax(ca.sumsqr(dx), min_rsr)**1.5)

qdot = vertcat(
    q[d*n:],
    vertcat(*vdot)*G
)

fn = ca.Function('f', [q], [qdot])
test = ca.DM(np.random.random(q.shape).tolist())

average_time = 0.0
for i in range(100):
    tic = perf_counter()
    fn(test)
    toc = perf_counter()
    print(i, toc-tic)
    average_time += (toc-tic) / 100
    
print(average_time)

intfunc = integrator('intfunc', 'rk', {'x': q, 'ode': qdot}, 0, np.linspace(0, tf, nt, endpoint=True, dtype=np.float32))
x0 = np.random.normal(0.0, 10.0, (d, n))
v0 = 0.1 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])@x0
q0 = x0.flatten('F').tolist()+v0.flatten('F').tolist()

tic = perf_counter()
res = intfunc(x0=q0)
toc = perf_counter()

print(toc-tic)
q = np.array(res['xf'].T).reshape((nt, 2*n, d))
# q = np.array(q0).reshape((1, 2*n, 2))
x = q[:, 0:n]
v = q[:, n:]

print(x.shape)

ax = plt.figure().add_subplot(projection='3d')
print(x[0, :, :].T.shape)
ax.scatter(*x[0, :, :].T)
for i in range(n): ax.plot(*x[:, i, :].T, label=f'Body {i}')
ax.legend()
plt.show()