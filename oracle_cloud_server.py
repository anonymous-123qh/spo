#!/usr/bin/env python3
import argparse
import numpy as np
import zmq




class OracleDemo:
    def __init__(self, dataset_path: str, state_is_next: bool = False):
        """
        dataset: npy where loaded object is something like [demo], demo is list of (state, action)
        state_is_next=False means: state == s_t for action a_t
        state_is_next=True  means: state == s_{t+1} for action a_t (we shift to make state-at-exec)
        """
        print(f"[CLOUD] Loading oracle demo from: {dataset_path}")
        task_data = np.load(dataset_path, allow_pickle=True)
        self.demos = []
        for d in range(len(task_data)):
            demo_steps = task_data[d]
            states = np.stack([np.asarray(x[0], np.float32) for x in demo_steps], axis=0)  # (T, state_dim)
            actions = np.stack([np.asarray(x[1], np.float32) for x in demo_steps], axis=0)  # (T, 8)
            self.demos.append((states, actions))
        demo = np.load(dataset_path, allow_pickle=True)[0]

        states = []
        actions = []
        for (s, a) in demo:
            states.append(np.asarray(s, dtype=np.float32))
            actions.append(np.asarray(a, dtype=np.float32))
        states = np.stack(states, axis=0)   # (T, state_dim)
        actions = np.stack(actions, axis=0) # (T, 8)

        if state_is_next:
            states = states[:-1]
            actions = actions[:-1]

        self.states = states
        self.actions = actions
        self.T = states.shape[0]
        self.state_dim = states.shape[1]
        self.action_dim = actions.shape[1]


class OracleCloud:
    def __init__(self, demo: OracleDemo, net_latency: float = 0.150, match_dims: int = 7):
        self.demo = demo
        self.net_latency = float(net_latency)
        self.match_dims = int(match_dims)
        self.current_idx = 0
        self.idx = {}  #current index
    def resync(self, actual_state: np.ndarray):
        """Nearest neighbor in joint space"""
        x = np.asarray(actual_state, dtype=np.float32).reshape(-1)
        d = self.match_dims
        # vectorized NN
        diffs = self.demo.states[:, :d] - x[:d]
        idx = int(np.argmin(np.linalg.norm(diffs, axis=1)))
        self.current_idx = idx

    def _get_idx(self, client_id: bytes):
        return self.idx.get(client_id, 0)

    def reset(self, client_id: bytes):
        self.idx[client_id] = 0
    def get_chunk(self, client_id: bytes, K: int):
        K = int(K)
        i0 = self._get_idx(client_id)
        i1 = min(i0 + K, self.demo.T)

        states = self.demo.states[i0:i1].copy()
        actions = self.demo.actions[i0:i1].copy()
        if actions.shape[1] >= 8:
            actions[:, 7] = (actions[:, 7] > 0.5).astype(np.float32)

        self.idx[client_id] = i1
        return actions, states


def pack_reply(actions: np.ndarray, states: np.ndarray):
    K = np.array([actions.shape[0]], dtype=np.int32)
    a_shape = np.array(actions.shape, dtype=np.int32)
    s_shape = np.array(states.shape, dtype=np.int32)
    return [
        K.tobytes(),
        a_shape.tobytes(),
        actions.astype(np.float32, copy=False).tobytes(),
        s_shape.tobytes(),
        states.astype(np.float32, copy=False).tobytes(),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--state-dim", type=int, required=True)
    ap.add_argument("--net-latency", type=float, default=0.150)
    ap.add_argument("--match-dims", type=int, default=7)
    ap.add_argument("--state-is-next", action="store_true",
                    help="Use if dataset stores (s_{t+1}, a_t) instead of (s_t, a_t)")
    args = ap.parse_args()

    demo = OracleDemo(args.dataset, state_is_next=args.state_is_next)
    if demo.state_dim != args.state_dim:
        raise ValueError(f"state_dim mismatch: dataset={demo.state_dim} vs arg={args.state_dim}")

    cloud = OracleCloud(demo, net_latency=args.net_latency, match_dims=args.match_dims)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(args.bind)

    print(f"[CLOUD] ROUTER bound at {args.bind}")
    print(f"[CLOUD] state_dim={args.state_dim} T={demo.T} net_latency={args.net_latency}s match_dims={args.match_dims}")
    print(f"[CLOUD] state_is_next={args.state_is_next}")

    while True:
        frames = sock.recv_multipart()

        client_id = frames[0]
        body = frames[1:]


        try:
            #request frames: [state(float32 bytes), K(int32 bytes), resync_flag(int32 bytes)]
            if len(body) != 3:
                raise ValueError(f"Bad request: expected 3 frames, got {len(body)}")

            #state = np.frombuffer(body[0], dtype=np.float32)
            K = int(np.frombuffer(body[1], dtype=np.int32)[0])
            reset_flag = int(np.frombuffer(body[2], np.int32)[0])

            if K <= 0:
                raise ValueError(f"Bad K: {K}")
            #resync_flag = int(np.frombuffer(body[2], dtype=np.int32)[0])

            #if state.size != args.state_dim:
            #    raise ValueError(f"Bad state size: got {state.size} expected {args.state_dim}")

            if reset_flag == 1:
                cloud.reset(client_id)

           #get the trajectory chunk
            actions, states = cloud.get_chunk(client_id, K)
            #pack the reply
            reply_frames = [client_id] + pack_reply(actions, states)

            #will estimate the latency from the edge
            #if args.net_latency > 0:
            #    time.sleep(args.net_latency)


            #oracle
            #if resync_flag == 1:
            #    cloud.resync(state)


            sock.send_multipart(reply_frames)

        except Exception as e:
            msg = str(e).encode("utf-8", errors="replace")
            sock.send_multipart([client_id, np.array([-1], np.int32).tobytes(), msg])
            print(f"[CLOUD][ERROR] {e}")


if __name__ == "__main__":
    main()
