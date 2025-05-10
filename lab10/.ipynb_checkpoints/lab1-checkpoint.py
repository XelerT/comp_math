import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from mpi4py import MPI

# mpirun -n 4 python lab1.py  --mode compare
# Running comparison with 4 processes for parallel part.
# Sequential Time (avg 3 runs): 0.068065 seconds
# Parallel Time (4 procs, avg 3 runs): 0.019937 seconds
# Speedup: 3.41
# Efficiency: 0.85


# u(t, 0) = psi(t), u(0, x) = phi(x)
def solve_transport_sequential(X, T, M, K, a, phi, psi, f):
    u = np.zeros((K + 1, M + 1))
    h = X / M
    tau = T / K

    for m in range(M + 1):
        u[0, m] = phi(m * h)

    for k in range(K):
        u[k + 1, 0] = psi((k + 1) * tau)
        for m in range(1, M):
             u[k + 1, m] = 0.5 * (u[k, m + 1] + u[k, m - 1]) - \
                           (tau / (2 * h)) * a * (u[k, m + 1] - u[k, m - 1]) + \
                           tau * f(k * tau, m * h)
        if M > 0:
           u[k + 1, M] = 2 * u[k + 1, M - 1] - u[k + 1, M - 2] if M > 1 else u[k+1, 0]

    return u


# u(t, 0) = psi(t), u(0, x) = phi(x)
def solve_transport_parallel(X, T, M, K, a, phi, psi, f, comm):
    size = comm.Get_size()
    rank = comm.Get_rank()

    h = X / M
    tau = T / K

    points_per_proc = M // size
    remainder = M % size
    local_M = points_per_proc + 1 if rank < remainder else points_per_proc
    local_start_m = rank * points_per_proc + min(rank, remainder)

    local_u = np.zeros((K + 1, local_M + 2)) # + 2 because of the ghost cell(assembling neighbors)

    for m_local_idx in range(local_M):
        m_global = local_start_m + m_local_idx
        local_u[0, m_local_idx + 1] = phi(m_global * h)

    for k in range(K):
        if rank == 0:
            local_u[k + 1, 0] = psi((k + 1) * tau)

        left_neighbor = rank - 1 if rank > 0 else MPI.PROC_NULL
        right_neighbor = rank + 1 if rank < size - 1 else MPI.PROC_NULL

        send_buf_right = local_u[k, local_M].copy()
        recv_buf_left = np.empty(1, dtype=local_u.dtype)
        comm.Sendrecv(send_buf_right, dest=right_neighbor, sendtag=0,
                      recvbuf=recv_buf_left, source=left_neighbor, recvtag=0)
        if left_neighbor != MPI.PROC_NULL:
             local_u[k, 0] = recv_buf_left[0]

        send_buf_left = local_u[k, 1].copy()
        recv_buf_right = np.empty(1, dtype=local_u.dtype)
        comm.Sendrecv(send_buf_left, dest=left_neighbor, sendtag=1,
                      recvbuf=recv_buf_right, source=right_neighbor, recvtag=1)
        if right_neighbor != MPI.PROC_NULL:
            local_u[k, local_M + 1] = recv_buf_right[0]

        if rank == 0:
             local_u[k, 0] = psi(k * tau)

        for m_local_idx in range(local_M):
            m_global = local_start_m + m_local_idx
            if m_global == 0 and rank == 0:
                 local_u[k + 1, m_local_idx + 1] = psi((k + 1) * tau)
                 continue
            if m_global == M:
                 continue

            if m_local_idx == 0 and rank == 0:
                 left_val = psi(k * tau)
            else:
                 left_val = local_u[k, m_local_idx]

            if m_local_idx == local_M - 1 and rank == size -1:
                 continue
            else:
                 right_val = local_u[k, m_local_idx + 2]

            center_val_prev_step_left = left_val
            center_val_prev_step_right = right_val

            local_u[k + 1, m_local_idx + 1] = 0.5 * (center_val_prev_step_right + center_val_prev_step_left) - \
                                           (tau / (2 * h)) * a * (center_val_prev_step_right - center_val_prev_step_left) + \
                                           tau * f(k * tau, m_global * h)

    local_result_data = local_u[K, 1:local_M+1].copy()
    sendcounts = np.array(comm.gather(local_M, root=0))
    displs = None
    total_M = M
    gathered_u_final = None
    if rank == 0:
        displs = np.concatenate(([0], np.cumsum(sendcounts[:-1])))
        gathered_u_final = np.empty(total_M, dtype=local_u.dtype)

    comm.Gatherv(sendbuf=local_result_data, recvbuf=(gathered_u_final, sendcounts, displs, MPI.DOUBLE), root=0)

    u_global_final = None
    if rank == 0:
        u_global_final = np.zeros(M + 1)
        u_global_final[0] = psi(T)
        u_global_final[1:M+1] = gathered_u_final

    return u_global_final


# u(t, 0) = psi(t), u(0, x) = phi(x)
def compare_times(X, T, M, K, a, phi, psi, f, num_runs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f"Running comparison with {size} processes for parallel part.")

    sequential_times = []
    if rank == 0:
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = solve_transport_sequential(X, T, M, K, a, phi, psi, f)
            end_time = time.perf_counter()
            sequential_times.append(end_time - start_time)
        avg_sequential_time = np.mean(sequential_times)
    else:
        avg_sequential_time = 0.0

    avg_sequential_time = comm.bcast(avg_sequential_time, root=0)

    parallel_times = []
    for _ in range(num_runs):
        comm.Barrier()
        start_time = MPI.Wtime()
        _ = solve_transport_parallel(X, T, M, K, a, phi, psi, f, comm)
        comm.Barrier()
        end_time = MPI.Wtime()
        if rank == 0:
            parallel_times.append(end_time - start_time)

    avg_parallel_time = np.mean(parallel_times) if rank == 0 else 0.0
    avg_parallel_time = comm.bcast(avg_parallel_time, root=0)

    if rank == 0:
        speedup = avg_sequential_time / avg_parallel_time if avg_parallel_time > 0 else float('inf')
        efficiency = speedup / size if avg_parallel_time > 0 else 0.0

        print(f"Sequential Time (avg {num_runs} runs): {avg_sequential_time:.6f} seconds")
        print(f"Parallel Time ({size} procs, avg {num_runs} runs): {avg_parallel_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}")
        print(f"Efficiency: {efficiency:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve 1D Transport Equation Sequentially or Parallelly')
    parser.add_argument('--mode', type=str, default='compare', choices=['seq', 'par', 'compare'],
                        help='Execution mode: sequential, parallel, compare times.')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs for averaging time in compare mode.')
    parser.add_argument('-M', type=int, default=100, help='Number of spatial steps.')
    parser.add_argument('-K', type=int, default=1000, help='Number of temporal steps.')
    parser.add_argument('-X', type=float, default=10.0, help='Spatial domain limit (0 <= x <= X)')
    parser.add_argument('-T', type=float, default=2.0, help='Temporal domain limit (0 <= t <= T)')
    parser.add_argument('-a', type=float, default=1.0, help='Transport coefficient.')

    args = parser.parse_args()

    X_val = args.X
    T_val = args.T
    M_val = args.M
    K_val = args.K
    a_val = args.a

    def phi_func(x):
        return np.exp(-((x - X_val / 4)**2) / 0.5)

    def psi_func(t):
        return 0.0

    def f_func(t, x):
        return 0.0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    u_to_plot = 1
    time_label = f"T = {T_val:.2f}"

    if args.mode == 'seq':
        if rank == 0:
            print(f"Running Sequential Solver (M={M_val}, K={K_val})...")
            start_t = time.perf_counter()
            u_result = solve_transport_sequential(X_val, T_val, M_val, K_val, a_val, phi_func, psi_func, f_func, comm=None)
            end_t = time.perf_counter()
            print(f"Sequential execution time: {end_t - start_t:.6f} seconds")
            if 'u_result' in locals():
                 u_to_plot = u_result[-1, :]

    elif args.mode == 'par':
         if rank == 0: print(f"Running Parallel Solver with {size} processes (M={M_val}, K={K_val})...")
         comm.Barrier()
         start_t = MPI.Wtime()
         u_result_par = solve_transport_parallel(X_val, T_val, M_val, K_val, a_val, phi_func, psi_func, f_func, comm)
         comm.Barrier()
         end_t = MPI.Wtime()
         if rank == 0:
            print(f"Parallel execution time: {end_t - start_t:.6f} seconds")
            if rank == 0 and 'u_result_par' in locals() and u_result_par is not None:
                u_to_plot = u_result_par
            
    elif args.mode == 'compare':
        compare_times(X_val, T_val, M_val, K_val, a_val, phi_func, psi_func, f_func, args.runs)

    if rank == 0 and u_to_plot is not None:
        x_coords = np.linspace(0, X_val, M_val + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, u_to_plot)
        plt.title(f'u(x) at {time_label}')
        plt.xlabel('x')
        plt.ylabel('u(T, x)')
        plt.grid(True)
        plt.ylim(np.min(u_to_plot) - 0.1, np.max(u_to_plot) + 0.1)

        plt.savefig("tep.png")
        plt.show()


    MPI.Finalize()
