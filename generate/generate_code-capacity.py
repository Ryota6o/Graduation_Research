import qldpc
from qldpc import codes
from qldpc.objects import Pauli
import stim
import sympy
import numpy as np
import yaml
import os
import json

# ==========================================
# 1. BBコード生成
# ==========================================
def make_bbcode(n0, k0, n1=None, k1=None, poly_a=None, poly_b=None):
    if n1 is None: n1 = n0
    if k1 is None: k1 = k0
    x, y = sympy.symbols('x y')
    if poly_a is None:
        poly_a = x**3 + y + y**2
    if poly_b is None:
        poly_b = y**3 + x + x**2
    code = codes.BBCode([n0, k0], poly_a=poly_a, poly_b=poly_b)
    return code

# ==========================================
# 2. Code Capacity 完全版シミュレーション
# ==========================================
def simulate_code_capacity_full(code, prob: float, num_shots: int):
    """
    XエラーとZエラーを両方正しく評価するために、2つの独立した実験を行い、
    結果を結合して「完全なCode Capacityデータ」を作成します。
    """
    n_qubits = code.num_qubits
    
    # --- Helper: Matrix to MPP targets ---
    def get_mpp_targets(matrix, pauli_type):
        """行列の行をStimのMPPターゲットリストに変換"""
        if not isinstance(matrix, np.ndarray): matrix = matrix.toarray()
        instructions = []
        for row in matrix:
            qubits = np.where(row)[0]
            if len(qubits) == 0: continue
            targets = []
            for q in qubits:
                if pauli_type == 'X': targets.append(stim.target_x(q))
                elif pauli_type == 'Z': targets.append(stim.target_z(q))
                targets.append(stim.target_combiner())
            if targets: targets.pop()
            instructions.append(targets)
        return instructions

    # -------------------------------------------------
    # Phase 1: Z基底シミュレーション (Xエラーの検出)
    # 初期状態: |0> (Stimデフォルト)
    # 測定: Zスタビライザー & 論理Z
    # -------------------------------------------------
    circ_z = stim.Circuit()
    circ_z.append("DEPOLARIZE1", range(n_qubits), prob)
    
    # Zスタビライザー測定
    if hasattr(code, 'matrix_z'):
        mpp_insts = get_mpp_targets(code.matrix_z, 'Z')
        for i, targets in enumerate(mpp_insts):
            circ_z.append("MPP", targets)
            circ_z.append("DETECTOR", [stim.target_rec(-1)], [i]) # IDは0から
    
    # 論理Z測定
    lz_start_idx = 0
    try:
        lz = code.get_logical_ops(Pauli.Z)
        mpp_insts = get_mpp_targets(lz, 'Z')
        for i, targets in enumerate(mpp_insts):
            circ_z.append("MPP", targets)
            circ_z.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], i)
            lz_start_idx += 1
    except: pass

    # -------------------------------------------------
    # Phase 2: X基底シミュレーション (Zエラーの検出)
    # 初期状態: |+> (RX命令で作成)
    # 測定: Xスタビライザー & 論理X
    # -------------------------------------------------
    circ_x = stim.Circuit()
    circ_x.append("RX", range(n_qubits)) # 全ビットを |+> に初期化
    circ_x.append("DEPOLARIZE1", range(n_qubits), prob)
    
    # Xスタビライザー測定
    if hasattr(code, 'matrix_x'):
        mpp_insts = get_mpp_targets(code.matrix_x, 'X')
        for i, targets in enumerate(mpp_insts):
            circ_x.append("MPP", targets)
            # Detector IDを変える必要はない(別々のサンプルとして扱うなら)が、
            # 結合する場合は配列上の位置で管理される。
            circ_x.append("DETECTOR", [stim.target_rec(-1)], [i]) 
    
    # 論理X測定
    try:
        lx = code.get_logical_ops(Pauli.X)
        mpp_insts = get_mpp_targets(lx, 'X')
        for i, targets in enumerate(mpp_insts):
            circ_x.append("MPP", targets)
            circ_x.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], i)
    except: pass

    # --- 実行 ---
    # それぞれ独立してサンプリング
    sampler_z = circ_z.compile_detector_sampler()
    dets_z, obs_z = sampler_z.sample(shots=num_shots, separate_observables=True)
    
    sampler_x = circ_x.compile_detector_sampler()
    dets_x, obs_x = sampler_x.sample(shots=num_shots, separate_observables=True)

    # --- データの結合 ---
    # シンドローム: [Z-stabs, X-stabs] の順に横結合
    # 論理エラー:   [Logical-Z-flips, Logical-X-flips] の順に横結合
    # これで、1つのショットの中に「Xエラー情報」と「Zエラー情報」の両方が含まれる形式になります
    
    full_detectors = np.concatenate([dets_z, dets_x], axis=1)
    full_observables = np.concatenate([obs_z, obs_x], axis=1)
    
    return full_detectors, full_observables

# ==========================================
# 3. メイン実行部
# ==========================================
def run_simulation_demo():
    config_path = "../config.yaml"
    default_config = {
        "DISTANCE": 6, "SHOTS": 1000000, "PROB": 0.01,
        "DATASET_DIR": "./datasets", "SEED": 43
    }
    # (Config読み込み省略...デフォルト値またはyamlを使用)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f: config = yaml.safe_load(f)
    else: config = default_config
        
    L = int(config.get("DISTANCE", 6))
    shots = int(config.get("SHOTS", 1000000))
    prob = float(config.get("PROB", 0.01))
    dataset_dir = config.get("DATASET_DIR", "./datasets")
    if config.get("SEED"): np.random.seed(config["SEED"])

    print(f"--- Full Code Capacity Simulation (X & Z) ---")
    
    try:
        M = L
        L = L*2
        code = make_bbcode(L, M)
    except Exception as e:
        print(f"Code gen error: {e}"); return

    print('ショット', shots, '確率', prob, ' データ・セットの場所',dataset_dir,'L,M',L,M)

    # 実行
    print("Sampling both X and Z basis...")
    dets, obs = simulate_code_capacity_full(code, prob, shots)
    print('dets',dets.shape)
    print('obs',obs.shape)
    # 統計
    # axis=1 (横方向) に1つでもTrueがあれば、そのショットは「論理エラー」
    # obsの中身がバイナリなら各行で１が含まれているのかをチエックする。
    failed_shots = np.sum(np.any(obs, axis=1))
    ler = failed_shots / shots

    print('--- 結果プレビュー ---')
    print(dets)
    print(f"Detectors shape: {dets.shape}") # [shots, num_checks_z + num_checks_x]
    print(obs)
    print(f"Observables shape: {obs.shape}") # [shots, k_z + k_x]
    print(f"Logical Error Rate: {ler:.4%} ({failed_shots}/{shots})")

    # ★★★ 追加: データの中身チェック ★★★
    obs_sum = np.sum(obs)
    det_sum = np.sum(dets)
    print(f"--- データの統計 ---")
    print(f"Total Observable Flips (Errors): {obs_sum} / {obs.size}")
    print(f"Error Rate: {obs_sum / obs.size:.4%}")
    print(f"Total Detection Events (Syndrome): {det_sum}")
    
    if obs_sum == 0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 致命的エラー: 生成されたデータにエラーが一つもありません !!!")
        print("!!!確率 p の設定か、シミュレーション関数を見直してください !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return # 保存せずに終了

    # 保存
    out_dir = os.path.expanduser(dataset_dir)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"bbcode_codecap_FULL_L{L}_p{prob:.3f}_shots{shots}.npz"
    save_path = os.path.join(out_dir, filename)
    
    print(f"Saving to {save_path}...")
    np.savez_compressed(
        save_path,
        L=np.array([L]), M=np.array([M]), p=np.array([prob]),
        noise_model=np.array(["code_capacity_full"]),
        label_y=dets.astype(np.float32),
        global_labels=obs.astype(np.float32),
        detectors=dets, observables=obs
    )
    
    # JSONメタデータ
    json_data = {
        "filename": filename,
        "type": "code_capacity_full",
        "description": "Merged Z-basis(X-err) and X-basis(Z-err) simulation.",
        "L": L, "M": M, "prob": prob, "shots": shots, "ler": float(ler)
    }
    with open(save_path + ".index.json", "w") as f: json.dump(json_data, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    run_simulation_demo()