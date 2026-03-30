# SPO
**Speculative Policy Orchestration: A Latency-Resilient Framework for Cloud-Robotic Manipulation**

## Overview
SPO is a cloud-edge execution framework for robotic manipulation in settings where policy and world-model inference are hosted remotely rather than on the robot itself. This deployment model is useful when the robot is compute-constrained, when multiple robots share a centralized model service, or when large policy/world models are too expensive to run fully on-board.

The challenge is that remote inference introduces network round-trip delay and jitter. In continuous manipulation, these delays can stall execution or make precomputed action chunks stale. SPO addresses this by combining speculative cloud-side rollout with local edge-side verification and execution.

This repository includes:

- a **cloud server** that serves action/state chunks from an oracle dataset
- an **edge client** that runs RLBench tasks and executes speculative actions
- support for multiple execution modes:
  - `spo`
  - `blocking`
  - `t1_sc`
  - `nftc`

## Installation

### Prerequisites
- Python 3.10
- `pip`
- RLBench installed on the **client machine**
- Two machines (cloud and client) with network connectivity such that the client can establish a connection to the cloud server and exchange data bidirectionally with it, for example:
  - on the same LAN
  - connected through a VPN such as Tailscale

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/anonymous-123qh/spo.git
   cd spo

2. Install packages:
   ```bash
    pip install -r requirements.txt
    ```


## Usage and Reproduce Results

### Running the Oracle World Model Experiment
### 1. Start the Cloud
**Syntax**
```bash
python3 oracle_cloud_server.py \
  --bind tcp://*:5555 \
  --dataset <world_model_dataset> \
  --state-dim <task_state_dimension> \
  --net-latency <network_delay_seconds>
```
**Example**
```bash
python3 oracle_cloud_server.py \
  --bind tcp://*:5555 \
  --dataset spo_dataset_v2_identical/StackBlocks_data.npy \
  --state-dim 148 \
  --net-latency 0.150
```
### 2. Start the edge client:
**Syntax**
```bash
python3 oracle_edge_client.py \
  --task <task_name> \
  --cloud tcp://<cloud_ip>:5555 \
  --headless \
  --hz <control_frequency> \
  --eps <verification_threshold>
```
**Example**
```bash
python3 oracle_edge_client.py \
  --task StackBlocks \
  --cloud tcp://127.0.0.1:5555 \
  --headless \
  --hz 50 \
  --eps 20
```
### Running the Trained World Model Experiment

### 1. Start the cloud server
The cloud server loads the trained world model and action model, then serves predicted trajectories to the edge client.

**Syntax**
```bash
python3 cloud_server.py \
  --bind tcp://*:5555 \
  --state-dim <task_state_dimension> \
  --world-model <world_model_path> \
  --action-model <action_model_path> \
  --net-latency <network_delay_seconds>
```
**Example**
```bash
python3 cloud_server.py \
--bind tcp://*:5555 \
--state-dim 148 \
--world-model identity_model/spo_cloud_model_StackBlocks.pth \
--action-model identity_model/spo_action_model_StackBlocks.pth \
--net-latency 0.150 \
```
### 2. Start the edge client:
**Syntax**
```bash
python3 edge_client.py \
  --task <task_name> \
  --cloud tcp://<cloud_ip>:5555 \
  --hz <control_frequency> \
  --headless
```
**Example**
```bash
python3 edge_client.py \
--task StackBlocks \
--cloud tcp://127.0.0.1:5555 \
--hz 50 \
--headless
```
## Demo Video



### StackBlocks

[![SPO StackBlocks demo](https://img.youtube.com/vi/IQnjq8ZxIrg/0.jpg)](https://www.youtube.com/watch?v=IQnjq8ZxIrg)

### InsertOntoSquarePeg
[![SPO InsertOntoSquarePeg demo](https://img.youtube.com/vi/XJIARVjhMj8/0.jpg)](https://www.youtube.com/watch?v=XJIARVjhMj8)

### PutAllGroceriesInCupboard
[![SPO PutAllGroceriesInCupboard demo](https://img.youtube.com/vi/GbojV6HF-90/0.jpg)](https://www.youtube.com/watch?v=GbojV6HF-90)
