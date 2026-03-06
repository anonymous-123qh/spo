#!/usr/bin/env python3
import argparse
import time
import numpy as np
import zmq
import torch
import torch.nn as nn


class SPOActionPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, s):
        return self.network(s)


class SPOWorldModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, state_dim),
        )

    def forward(self, s):
        return self.network(s)


class SPOCloudNode:
    def __init__(self, state_dim, world_model_path, action_model_path, net_latency=0.150, device="cpu"):
        self.state_dim = int(state_dim)
        self.net_latency = float(net_latency)
        self.device = torch.device(device)

        self.world_model = SPOWorldModel(self.state_dim).to(self.device)
        self.action_policy = SPOActionPolicy(self.state_dim).to(self.device)

        self.world_model.load_state_dict(torch.load(world_model_path, map_location=self.device, weights_only=True))
        self.action_policy.load_state_dict(torch.load(action_model_path, map_location=self.device, weights_only=True))

        self.world_model.eval()
        self.action_policy.eval()

    def request_trajectory(self, current_state_np: np.ndarray, K: int):
        """Return actions (K,8), states (K,state_dim)"""
        if self.net_latency > 0:
            time.sleep(self.net_latency)

        s_np = np.asarray(current_state_np, dtype=np.float32).reshape(-1)
        if s_np.size != self.state_dim:
            raise ValueError(f"state_dim mismatch: got {s_np.size}, expected {self.state_dim}")

        K = int(K)
        actions = np.zeros((K, 8), dtype=np.float32)
        states = np.zeros((K, self.state_dim), dtype=np.float32)

        s = torch.tensor(s_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for i in range(K):
                #cache the state that expect the robot to be at when executing this action
                s_np_i = s.detach().cpu().numpy().astype(np.float32)
                states[i] = s_np_i

                #predict the action to take at this state
                a = self.action_policy(s)
                a_np = a.detach().cpu().numpy().astype(np.float32)
                a_np[7] = 1.0 if a_np[7] > 0.5 else 0.0  #clamp gripper
                actions[i] = a_np

                #predict the next state
                s = self.world_model(s)

        return actions, states


def pack_reply(actions: np.ndarray, states: np.ndarray):

    K = np.array([actions.shape[0]], dtype=np.int32)
    a_shape = np.array(actions.shape, dtype=np.int32)  # (K,8)
    s_shape = np.array(states.shape, dtype=np.int32)   # (K,state_dim)

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
    ap.add_argument("--state-dim", type=int, required=True)
    ap.add_argument("--world-model", required=True)
    ap.add_argument("--action-model", required=True)
    ap.add_argument("--net-latency", type=float, default=0.150)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cloud = SPOCloudNode(
        state_dim=args.state_dim,
        world_model_path=args.world_model,
        action_model_path=args.action_model,
        net_latency=args.net_latency,
        device=args.device,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.ROUTER)
    #sock.setsockopt(zmq.TCP_NODELAY, 1)
    sock.bind(args.bind)

    print(f"[CLOUD] ROUTER bound at {args.bind}")
    print(f"[CLOUD] state_dim={args.state_dim} device={args.device} net_latency={args.net_latency}s")

    while True:
        frames = sock.recv_multipart()
        #prepends client identity frame
        client_id = frames[0]
        body = frames[1:]

        try:
            if len(body) != 2:
                raise ValueError(f"Bad request: expected 2 frames, got {len(body)}")

            state = np.frombuffer(body[0], dtype=np.float32)
            K = int(np.frombuffer(body[1], dtype=np.int32)[0])

            actions, states = cloud.request_trajectory(state, K)
            reply = pack_reply(actions, states)
            sock.send_multipart([client_id] + reply)

        except Exception as e:
            #error reply: K=-1 + message
            msg = str(e).encode("utf-8", errors="replace")
            sock.send_multipart([client_id, np.array([-1], np.int32).tobytes(), msg])
            print(f"[CLOUD][ERROR] {e}")


if __name__ == "__main__":
    main()