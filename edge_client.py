#!/usr/bin/env python3
import argparse
import time
import threading
import queue
import numpy as np
import random
import zmq
import os
import csv

from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import PutAllGroceriesInCupboard, StackBlocks, InsertOntoSquarePeg
from rlbench.observation_config import ObservationConfig


TASKS = {
    "PutAllGroceriesInCupboard": PutAllGroceriesInCupboard,
    "StackBlocks": StackBlocks,
    "InsertOntoSquarePeg": InsertOntoSquarePeg,
}

TASK_DIMS = {
    "PutAllGroceriesInCupboard": 295,
    "StackBlocks": 148,
    "InsertOntoSquarePeg": 141,
}


def ahs_update(K, e_miss, eps_base, K_min, K_max, beta=1):
    if e_miss > 0:
        rho = max(e_miss / max(eps_base, 1e-6), 1e-6)
        newK = int(np.floor(K / rho))
        return max(K_min, newK)
    else:
        return min(K_max, K + beta)


class SPOEdgeNode:
    def __init__(self, eps_base=1.5):
        self.action_cache = []
        self.state_cache = []
        self.eps_base = float(eps_base)

    def cache_size(self):
        return len(self.state_cache)

    def flush(self):
        self.action_cache = []
        self.state_cache = []

    def fill_cache(self, actions, states):
        self.action_cache = [actions[i].copy() for i in range(actions.shape[0])]
        self.state_cache = [states[i].copy() for i in range(states.shape[0])]

    def verify_and_pop(self, actual_state):
        if len(self.state_cache) == 0:
            return None, 0.0

        expected = self.state_cache[0]
        e = float(np.linalg.norm(actual_state[:7] - expected[:7]))

        if e <= self.eps_base:
            a = self.action_cache.pop(0)
            self.state_cache.pop(0)
            return a, e
        else:
            self.flush()
            return None, e


def extract_padded_state(obs, expected_dim):
    raw = np.concatenate([
        obs.joint_positions,
        [1.0 if obs.gripper_open else 0.0],
        obs.task_low_dim_state
    ]).astype(np.float32)

    if raw.size < expected_dim:
        return np.concatenate([raw, np.zeros(expected_dim - raw.size, np.float32)]).astype(np.float32)
    if raw.size > expected_dim:
        return raw[:expected_dim].astype(np.float32)
    return raw


def make_hold_action_from_obs(obs):
    grip = 1.0 if obs.gripper_open else 0.0
    return np.concatenate([obs.joint_positions.astype(np.float32), np.array([grip], np.float32)])


def unpack_reply(frames):
    if len(frames) == 2:
        K = int(np.frombuffer(frames[0], np.int32)[0])
        msg = frames[1].decode("utf-8", errors="replace")
        return K, None, None, msg

    if len(frames) != 5:
        return -1, None, None, f"Bad reply frame count: {len(frames)}"

    K = int(np.frombuffer(frames[0], np.int32)[0])
    a_shape = tuple(np.frombuffer(frames[1], np.int32))
    actions = np.frombuffer(frames[2], np.float32).reshape(a_shape)

    s_shape = tuple(np.frombuffer(frames[3], np.int32))
    states = np.frombuffer(frames[4], np.float32).reshape(s_shape)

    return K, actions, states, None


class ZMQAsyncRequester:
    """
    Network thread owns the DEALER socket
    Control loop never blocks... it polls reply_queue
    """
    def __init__(self, cloud_addr: str):
        self.cloud_addr = cloud_addr
        self.req_q = queue.Queue(maxsize=1)
        self.rep_q = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.DEALER)
        sock.connect(self.cloud_addr)

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while not self.stop.is_set():
            try:
                state_np, K = self.req_q.get_nowait()
                sock.send_multipart([
                    np.asarray(state_np, np.float32).tobytes(),
                    np.array([int(K)], np.int32).tobytes()
                ])
            except queue.Empty:
                pass

            events = dict(poller.poll(timeout=1))
            if sock in events and events[sock] & zmq.POLLIN:
                frames = sock.recv_multipart()
                while True:
                    try:
                        self.rep_q.get_nowait()
                    except queue.Empty:
                        break
                self.rep_q.put(frames)

    def request(self, state_np, K):
        while True:
            try:
                self.req_q.get_nowait()
            except queue.Empty:
                break
        try:
            self.req_q.put_nowait((state_np, int(K)))
        except queue.Full:
            pass

    def poll_reply(self):
        try:
            return self.rep_q.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self):
        self.stop.set()
        self.t.join(timeout=1.0)


def run(run_id, method, task_name, cloud_addr, max_steps, control_hz, eps_base, K_min, K_max, beta, headless):
    task_class = TASKS[task_name]
    state_dim = TASK_DIMS[task_name]
    dt = 1.0 / float(control_hz)


    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_open = True
    obs_config.task_low_dim_state = True

    action_mode = MoveArmThenGripper(JointPosition(), Discrete())
    env = Environment(action_mode, obs_config=obs_config, dataset_root="", headless=headless)
    env.launch()
    task = env.get_task(task_class)

    np.random.seed(42 + run_id)
    random.seed(42 + run_id)
    _, obs = task.reset()

    edge = SPOEdgeNode(eps_base=eps_base)
    net = ZMQAsyncRequester(cloud_addr)

    K = int(K_min)
    inflight = False

    hits = 0
    misses = 0
    deadline_misses = 0
    hold_time = 0.0
    start = time.time()

    cloud_requests = 0
    cloud_replies_ok = 0
    cache_steps_since_reply = 0
    cache_steps_per_reply = []

    step_records = []
    task_success = False
    reward = 0.0

    for step in range(max_steps):
        tick = time.time()

        rep = net.poll_reply()
        if rep is not None:
            K_r, actions, states, err = unpack_reply(rep)
            if K_r > 0 and actions is not None:
                cache_steps_per_reply.append(cache_steps_since_reply)
                cache_steps_since_reply = 0
                edge.fill_cache(actions, states)
                cloud_replies_ok += 1
            inflight = False

        actual_state = extract_padded_state(obs, expected_dim=state_dim)
        speculative_action, e_t = edge.verify_and_pop(actual_state)

        cache_empty = (edge.cache_size() == 0)

        #baselines
        if method == "blocking":
            K = 1
        elif method == "t1_sc":
            K = 2
        elif method == "nftc":
            K = 10
        else:  # spo
            if speculative_action is not None:
                K = ahs_update(K, e_miss=0.0, eps_base=eps_base, K_min=K_min, K_max=K_max, beta=beta)
            else:
                e_miss = e_t if e_t > 0 else 0.0
                K = ahs_update(K, e_miss=e_miss, eps_base=eps_base, K_min=K_min, K_max=K_max, beta=beta)

        if speculative_action is not None:
            action = speculative_action
            hits += 1
            cache_steps_since_reply += 1
            is_hit = 1
        else:
            misses += 1
            is_hit = 0

            action = make_hold_action_from_obs(obs)
            hold_time += dt

            if not inflight:
                net.request(actual_state, K)
                cloud_requests += 1
                inflight = True

        step_records.append({
            "method": method,
            "task": task_name,
            "run_id": run_id,
            "step": step,
            "target_k": K,
            "cache_size": edge.cache_size(),
            "e_t": round(e_t, 5),
            "is_hit": is_hit,
            "cumulative_hold_time": round(hold_time, 4),
            "cumulative_requests": cloud_requests,
            "inflight": 1 if inflight else 0,
            "cache_empty": 1 if cache_empty else 0,
        })

        if step % 20 == 0:
            print(
                f"Step {step:03d} | method={method} | e={e_t:.4f} | "
                f"cache={edge.cache_size():02d} | K={K:02d} | inflight={inflight}"
            )

        try:
            obs, reward, terminate = task.step(action)
        except Exception:
            _, obs = task.reset()
            continue

        if terminate:
            print(f"Task completed at step {step}")
            task_success = True
            break

        elapsed = time.time() - tick
        sleep_t = dt - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            deadline_misses += 1

    net.shutdown()
    env.shutdown()

    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)

    #save perstep traces
    if step_records:
        filename = f"results_{method}_{task_name}_eps{eps_base}_run{run_id}.csv"
        csv_path = os.path.join(log_dir, filename)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=step_records[0].keys())
            writer.writeheader()
            writer.writerows(step_records)
        print(f"Saved step-level traces to {csv_path}")

    total = hits + misses
    hit_rate = 100.0 * hits / total if total else 0.0
    wall = time.time() - start
    avg_cache_steps = (sum(cache_steps_per_reply) / len(cache_steps_per_reply)) if cache_steps_per_reply else 0.0
    max_cache_steps = max(cache_steps_per_reply) if cache_steps_per_reply else 0
    min_cache_steps = min(cache_steps_per_reply) if cache_steps_per_reply else 0

    print("\n" + "=" * 60)
    print(f"EDGE RESULTS -- Task {task_name} -- Method {method}")
    print("=" * 60)
    print(f"Steps            : {total}")
    print(f"Hit rate         : {hit_rate:.2f}%")
    print(f"Hold time (est)  : {hold_time:.2f} s")
    print(f"Deadline misses  : {deadline_misses} (>{dt*1000:.1f} ms)")
    print(f"Wall clock       : {wall:.2f} s")
    print(f"Cloud requests   : {cloud_requests}")
    print(f"Cloud replies ok : {cloud_replies_ok}")
    print(f"Cache steps/reply: avg={avg_cache_steps:.2f} min={min_cache_steps} max={max_cache_steps}")
    print("=" * 60)

    #save per-run summary
    summary_file = os.path.join(log_dir, "experiment_summary.csv")
    file_exists = os.path.isfile(summary_file)

    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "method", "task", "run_id", "success",
                "items_stowed", "total_steps", "wall_clock_time", "idle_time"
            ])

        writer.writerow([
            method,
            task_name,
            run_id,
            int(task_success),
            reward,
            total,
            round(wall, 2),
            round(hold_time, 2)
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=1, help="Trial number for logging")
    ap.add_argument("--method", choices=["spo", "blocking", "t1_sc", "nftc"], default="spo")
    ap.add_argument("--task", choices=TASKS.keys(), default="StackBlocks")
    ap.add_argument("--cloud", required=True, help="e.g. tcp://[YOUR-CLOUD-IP-HERE]:5555")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1.5)
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--beta", type=int, default=1)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    run(
        run_id=args.run_id,
        method=args.method,
        task_name=args.task,
        cloud_addr=args.cloud,
        max_steps=args.max_steps,
        control_hz=args.hz,
        eps_base=args.eps,
        K_min=args.kmin,
        K_max=args.kmax,
        beta=args.beta,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()