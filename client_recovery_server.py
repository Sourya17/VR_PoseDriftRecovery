import time
import socket
import threading
import numpy as np
import torch
import pickle
import json
import pandas as pd
from collections import deque, defaultdict
from typing import List, Dict
import zmq
import random
import matplotlib.pyplot as plt
import signal
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R, Slerp

mse_series = []
P_DROP = 0.7


# === Anomaly classification counters ===
stats = {
    "total": 0,
    "normal": 0,
    "soft": 0,
    "hard": 0
}

recovery_state = {
    "consecutive_hard": 0,
    "recovery_triggered": 0
}


DISABLE_CORRECTION = False  # Set to False to enable filtering and anomaly correction
# Set to True to record MSE of unrecovered poses


# =============================================================================== #

# === Kalman filters ===
class SimpleKalman1D:
    def __init__(self, q=1e-2, r=0.1, p=1.0, x=0.0):
        self.q = q; self.r = r; self.p = p; self.x = x
    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

class KalmanVector3D:
    def __init__(self, q=1e-3, r=0.05):
        self.filters = [SimpleKalman1D(q=q, r=r) for _ in range(3)]
    def update(self, vec):
        return np.array([f.update(v) for f, v in zip(self.filters, vec)], dtype=np.float32)

pose_filters = {
    "pos": KalmanVector3D(q=1e-5, r=0.01),
    "vel": KalmanVector3D(q=1e-4, r=0.02),
    "ba":  KalmanVector3D(q=1e-5, r=0.01),
    "bg":  KalmanVector3D(q=1e-5, r=0.01)
}
quat_history = deque(maxlen=5)

# ====================================================================================== #
# --------------- DEFINED BUT NOT USED ------------------------------ #




# === Load AE, Scaler, PCA, and thresholds ===
load_start = time.time()
scaler = pickle.load(open("plots_exaple/ae_parameters_multi/scaler.pkl", "rb"))
pca = pickle.load(open("plots_exaple/ae_parameters_multi/pca.pkl", "rb"))
ae = torch.jit.load("plots_exaple/ae_parameters_multi/ae.pt").eval()
with open("plots_exaple/ae_parameters_multi/cols.json", "r") as f:
    cols = json.load(f)
with open("plots_exaple/ae_parameters_multi/thresholds.json", "r") as f:
    thresholds = json.load(f)

pose_model = torch.jit.load("plots_exaple/pose_predictor/pose_predictor.pt").eval()
pose_cols = cols




THR_P98 = thresholds["thr_p98"]
THR_MAD = thresholds["thr_mad"]
THR_P99 = thresholds["thr_p99"]
print(f"[TIMING] Model and scaler load time: {(time.time() - load_start)*1000:.2f} ms")

buf_lock = threading.Lock()
fast_ring = deque(maxlen=5000)
imu_ring = defaultdict(list)

# === ZMQ Setup ===
print("[ZMQ] Initializing ZMQ context and sockets...")
context = zmq.Context()
fast_sock = context.socket(zmq.SUB)
fast_sock.connect("tcp://127.0.0.1:5556")
print("[ZMQ] Connected to FAST on tcp://127.0.0.1:5556")
fast_sock.setsockopt_string(zmq.SUBSCRIBE, "")

imu_sock = context.socket(zmq.SUB)
imu_sock.connect("tcp://127.0.0.1:5557")
print("[ZMQ] Connected to IMU on tcp://127.0.0.1:5557")
imu_sock.setsockopt_string(zmq.SUBSCRIBE, "")

# === Feature Engineering Helpers === 
def safe_mom(x):
    x = np.asarray(x, float)
    if x.size == 0 or np.isnan(x).any():
        return (0., 0., 0., 0., 0., 0.)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.
    mn, mx = float(x.min()), float(x.max())
    xc = x - mean
    m2 = np.mean(xc ** 2)
    m3 = np.mean(xc ** 3)
    m4 = np.mean(xc ** 4)
    skew = m3 / (m2 ** 1.5 + 1e-12)
    kurt = m4 / (m2 ** 2 + 1e-12) - 3.0
    return (mean, std, mn, mx, skew, kurt)

def add_comp(prefix, arr, feat, cols_out):
    arr = np.zeros((0, 3)) if arr.size == 0 else arr
    for j in range(arr.shape[1]):
        for k, v in zip(("mean", "std", "min", "max", "skew", "kurt"), safe_mom(arr[:, j])):
            feat.append(v)
            cols_out.append(f"{prefix}_{j}_{k}")

def add_scal(prefix, x, feat, cols_out):
    for k, v in zip(("mean", "std", "min", "max", "skew", "kurt"), safe_mom(x)):
        feat.append(v)
        cols_out.append(f"{prefix}_{k}")

def quat_angle_deg(q1, q2):
    v = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2 * np.degrees(np.arccos(v))

def build(prev_slow, slow, fasts, imu):
    build_start = time.time()
    fts = np.array([f["ts"] for f in fasts], np.int64)
    idx = np.nonzero((fts >= prev_slow["ts"]) & (fts < slow["ts"]))[0]
    if idx.size == 0:
        return pd.DataFrame(), []

    fw = [fasts[j] for j in idx]
    feat, cols_out = [], []

    dpos = np.stack([f["pos"] - slow["pos"] for f in fw])
    add_comp("dpos", dpos, feat, cols_out)
    add_scal("dpos_norm", np.linalg.norm(dpos, 1), feat, cols_out)

    dvel = np.stack([f["vel"] - slow["vel"] for f in fw])
    add_comp("dvel", dvel, feat, cols_out)
    add_scal("dvel_norm", np.linalg.norm(dvel, 1), feat, cols_out)

    add_scal("q_angle_deg", [quat_angle_deg(f["quat"], slow["quat"]) for f in fw], feat, cols_out)

    for key in ("b_acc", "b_gyro", "b_acc_prev", "b_gyro_prev"):
        add_comp(key, np.stack([f[key] for f in fw]), feat, cols_out)

    iw, ia = [], []
    for f in fw:
        for sm in imu.get(int(f["ts"]), []):
            iw.append(sm["w"])
            ia.append(sm["a"])
    add_comp("imu_w", np.stack(iw) if iw else np.zeros((0,3)), feat, cols_out)
    add_scal("imu_w_norm", np.linalg.norm(iw, 1) if iw else np.zeros(0), feat, cols_out)
    add_comp("imu_a", np.stack(ia) if ia else np.zeros((0,3)), feat, cols_out)
    add_scal("imu_a_norm", np.linalg.norm(ia, 1) if ia else np.zeros(0), feat, cols_out)

    for k in ("pos","vel","ba","bg"):
        for j,v in enumerate(slow[k]):
            feat.append(float(v))
            cols_out.append(f"slow_{k}_{j}")

    feat.insert(0, (slow["ts"] - prev_slow["ts"]) / 1e9)
    cols_out.insert(0, "win_dur_s")
    feat.insert(0, float(len(idx)))
    cols_out.insert(0, "win_n_fast")
    print(f"[TIMING] build() duration: {(time.time() - build_start) * 1000:.2f} ms")

    return pd.DataFrame([feat], columns=cols_out), cols_out


def predict_noise(df):
    X = df[[c for c in df.columns if c in cols]].to_numpy(dtype=np.float32)
    Z = pca.transform(scaler.transform(X))
    with torch.no_grad():
        pred_scaled = noise_model(torch.tensor(Z, dtype=torch.float32)).numpy()
    return pred_scaled * y_scale  # shape (N, 16)






# === FAST & IMU Threads ===
def fast_thread():
    while True:
        msg = fast_sock.recv_string()
        parts = msg.strip().split(",")
        if parts[0] != "[FAST]": continue
        ts = int(parts[1])
        vals = list(map(float, parts[2:]))
        with buf_lock:
            fast_ring.append({
                "ts": ts,
                "pos": np.array(vals[0:3], dtype=np.float32),
                "vel": np.array(vals[3:6], dtype=np.float32),
                "quat": np.array(vals[6:10], dtype=np.float32),
                "b_acc": np.array(vals[10:13], dtype=np.float32),
                "b_gyro": np.array(vals[13:16], dtype=np.float32),
                "b_acc_prev": np.array(vals[16:19], dtype=np.float32),
                "b_gyro_prev": np.array(vals[19:22], dtype=np.float32),
            })

def imu_thread():
    while True:
        msg = imu_sock.recv_string()
        parts = msg.strip().split(",")
        if parts[0] != "[IMU]": continue
        ts = int(parts[1])
        vals = list(map(float, parts[2:]))
        with buf_lock:
            imu_ring[ts].append({
                "w": np.array(vals[0:3], dtype=np.float32),
                "a": np.array(vals[3:6], dtype=np.float32),
            })


# === TCP Server Handler ===
def handler(conn):
    buf = ""
    print("[TCP] Client connected.")
    with conn:
        while True:
            data = conn.recv(8192)
            if not data:
                break
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                parts = line.strip().split(",")
                if parts[0] != "[SLOW]":
                    continue

                total_start = time.time()

                ts = int(parts[1])
                stats["total"] += 1
                vals = list(map(float, parts[2:]))
                slow = {
                    "ts": ts,
                    "ba": np.array(vals[0:3], dtype=np.float32),
                    "bg": np.array(vals[3:6], dtype=np.float32),
                    "pos": np.array(vals[6:9], dtype=np.float32),
                    "vel": np.array(vals[9:12], dtype=np.float32),
                    "quat": np.array(vals[12:16], dtype=np.float32),
                }

                if not hasattr(handler, "prev_slow"):
                    handler.prev_slow = None

                if handler.prev_slow is None:
                    corrected = slow
                    print(f"[SLOW POSE] ts={ts} -> first pose")
                else:
                    with buf_lock:
                        fasts = list(fast_ring)
                        imu = dict(imu_ring)
                    df, _ = build(handler.prev_slow, slow, fasts, imu)
                    if df.empty:
                        corrected = slow
                    else:
                        prep_start = time.time()
                        df_filtered = df[[c for c in df.columns if c in cols and not any(k in c for k in ("skew", "kurt", "slow_pos", "slow_vel", "win_n_fast", "win_dur_s"))]]
                        X = df_filtered.to_numpy(dtype=np.float32)
                        X = scaler.transform(X)
                        X = np.clip(X, -6.0, 6.0)
                        Z = pca.transform(X)
                        prep_time = (time.time() - prep_start) * 1000

                        infer_start = time.time()
                        with torch.no_grad():
                            Z_recon = ae(torch.tensor(Z, dtype=torch.float32))
                        infer_time = (time.time() - infer_start) * 1000

                        mse = torch.mean((torch.tensor(Z, dtype=torch.float32) - Z_recon) ** 2, dim=1).item()

                        mse_series.append(mse)

                        print(f"[TIMING] preproc+PCA: {prep_time:.2f} ms")
                        print(f"[TIMING] AE inference: {infer_time:.2f} ms")

                        '''
                        if mse > THR_P98:
                            conn.sendall(b"-9999\n")
                            print(f"[ANOMALY] ts={ts} → HARD anomaly (MSE={mse:.4f})")
                            handler.prev_slow = slow
                            stats["hard"] += 1
                            print(f"[TIMING] total: {(time.time() - total_start)*1000:.2f} ms")
                            return
                        '''
                        if DISABLE_CORRECTION:
                            # Skip classification & correction, return unmodified slow pose
                            print(f"[BYPASS MODE] ts={ts} -> MSE={mse:.4f} (returning raw slow pose)")
                            vec16 = np.hstack([
                                slow["pos"], slow["quat"], slow["ba"], slow["bg"],
                                slow["pos"], slow["vel"], slow["quat"]
                            ]).astype(np.float32)
                            out = " ".join(f"{v:.6f}" for v in vec16) + "\n"
                            conn.sendall(out.encode())
                            handler.prev_slow = slow
                            return

                        
                        
                        
                        if mse > THR_P98:
                            recovery_state["consecutive_hard"] += 1

                            if recovery_state["consecutive_hard"] >= 12:
                                
                                vec16 = np.hstack([
                                    slow["pos"], slow["quat"], slow["ba"], slow["bg"],
                                    slow["pos"], slow["vel"], slow["quat"]
                                ]).astype(np.float32)
                                
                                out = " ".join(f"{v:.6f}" for v in vec16) + "\n"
                                conn.sendall(out.encode())
                                print(f"[RECOVERY] ts={ts} -> forced pass after 12 hard anomalies (MSE={mse:.4f})")
                                stats["normal"] += 1
                                recovery_state["recovery_triggered"] += 1
                                recovery_state["consecutive_hard"] = 0
                                handler.prev_slow = slow
                                print(f"[TIMING] total: {(time.time() - total_start)*1000:.2f} ms")
                                return
                            else:
                                conn.sendall(b"-9999\n")
                                print(f"[ANOMALY] ts={ts} -> HARD anomaly (MSE={mse:.4f})")
                                stats["hard"] += 1
                                handler.prev_slow = slow
                                print(f"[TIMING] total: {(time.time() - total_start)*1000:.2f} ms")
                                return
                                
                        elif mse > THR_MAD:
                            
                            #####################################################################################
                            conn.sendall(b"-9999\n")
                            handler.prev_slow = slow
                            print(f"[ANOMALY] ts={ts} → SOFT anomaly (MSE={mse:.4f})")
                            stats["soft"] += 1
                            return

                        else:
                            corrected = slow
                            ##############################################################
                            print(f"[OK] ts={ts} -> NORMAL (MSE={mse:.4f})")
                            stats["normal"] += 1
                            recovery_state["consecutive_hard"] = 0


                handler.prev_slow = corrected
                vec16 = np.hstack([
                    corrected["pos"], corrected["quat"], corrected["ba"], corrected["bg"],
                    corrected["pos"], corrected["vel"], corrected["quat"]
                ]).astype(np.float32)
                out = " ".join(f"{v:.6f}" for v in vec16) + "\n"
                conn.sendall(out.encode())

                print(f"[TIMING] total: {(time.time() - total_start)*1000:.2f} ms")
                
                
# === Launch TCP Server ===
def tcp_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", 5555))
    srv.listen()
    print("[SERVER] Listening on port 5555...")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handler, args=(conn,), daemon=True).start()

# === Graceful Exit ===
def graceful_exit(sig, frame):
    print("\n[CTRL+C] Shutting down. Saving MSE plot...")
    if mse_series:
        pd.Series(mse_series).to_csv("plots_exaple/mse_series.csv", index=False)
        print("[DONE] MSE values saved to mse_series.csv")
        plt.figure(figsize=(8, 4))
        plt.plot(mse_series)
        plt.title("MSE Over Time")
        plt.xlabel("Frame Index")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots_exaple/mse_plot.png")
        print("[DONE] Plot saved as mse_plot.png")
    else:
        print("[WARNING] No MSE data to save.")
    print("\n=== Summary ===")
    print(f"Total slow poses processed: {stats['total']}")
    print(f"  - NORMAL: {stats['normal']}")
    print(f"  - SOFT  : {stats['soft']}")
    print(f"  - HARD  : {stats['hard']}")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, graceful_exit)
    threading.Thread(target=fast_thread, daemon=True).start()
    threading.Thread(target=imu_thread, daemon=True).start()
    threading.Thread(target=tcp_server, daemon=True).start()
    print("[MAIN] Server running. Press Ctrl+C to exit and save plot.")
    signal.pause()

