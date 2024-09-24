import open3d as o3d
import numpy as np
from numba import cuda

a = 200
b = 30
n = a*b


data = np.random.normal(size=(3, n)).astype(np.float32)
x = cuda.to_device(data[0])
y = cuda.to_device(data[1])
z = cuda.to_device(data[2]*0.1)
vx = cuda.to_device(1000*np.sign(data[1])*np.sqrt(np.abs(data[1]))*(0.95+0.1*np.random.random(data[1].shape)))
vy = cuda.to_device(-1000*np.sign(data[0])*np.sqrt(np.abs(data[0]))*(0.95+0.1*np.random.random(data[1].shape)))
vz = cuda.to_device(data[2])

dt = 0.0001
@cuda.jit
def cudakernel2(x, y, z, vx, vy, vz):
    thread_position = cuda.grid(1)

    curx = x[thread_position]
    cury = y[thread_position]
    curz = z[thread_position]
    for i in range(n):
        if i==thread_position: continue
        scl = (max((curx - x[i])**2 + (cury - y[i])**2 + (curz - z[i])**2, 0.1))**1.5
        vx[thread_position] += -scl*(curx-x[i])*dt
        vy[thread_position] += -scl*(cury-y[i])*dt
        vz[thread_position] += -scl*(curz-z[i])*dt
    x[thread_position] += vx[thread_position]*dt
    y[thread_position] += vy[thread_position]*dt
    z[thread_position] += vz[thread_position]*dt


pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(
    np.vstack([
        x.copy_to_host(),
        y.copy_to_host(),
        z.copy_to_host()
    ]).T.astype(np.float64)
)
pc.paint_uniform_color([0.95, 0.5, 0.3])

def run_viz():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)

    while True:
        cudakernel2[a, b](x, y, z, vx, vy, vz)
        pc.points = o3d.utility.Vector3dVector(
            np.vstack([
                x.copy_to_host(),
                y.copy_to_host(),
                z.copy_to_host()
            ]).T.astype(np.float64)
        )
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()

if __name__ == '__main__':
    run_viz()