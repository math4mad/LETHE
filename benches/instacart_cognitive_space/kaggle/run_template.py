import os, sys, json, base64, gzip, time, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 基础设施 (Infrastructure) ---
_osx = os.system # 兼容层

# 疫苗 (Vaccine)
_osx("python -m pip uninstall -y -q torchao 2>/dev/null")

# --- 旋钮 (Knobs) ---
SEED = 13
PROBE_SAMPLES = 200
INTENT_SAMPLES = 100
MODEL_SAMPLES = 500 # 每类抽样数
SAMPLES_PER_ORDER = 10 # 每个订单限制 token 数防止 OOM

# 正本内嵌占位 (Placeholder)
# CORPUS_B64 = "#__CORPUS_B64__#"
# QUANT = "#__QUANT__#"

# --- 核心引擎 (Engine) ---

def load_embedded_corpus(b64s):
    try:
        decompressed = gzip.decompress(base64.b64decode(b64s))
        return json.loads(decompressed.decode('utf-8'))
    except Exception as e:
        print(f"Corpus load error: {e}")
        return None

def get_centroid_acc(hidden_states, labels, num_classes):
    """
    Centroid Probing: 计算质心距离分类准确率
    hidden_states: [N, D]
    labels: [N] (0 to num_classes-1)
    """
    # 关键修复: 统一 dtype —— 模型输出在 bf16 (bf16/4bit 加载), 而质心累加与余弦相似度的
    # 矩阵乘必须在同一 dtype, 否则 torch.mm(bf16, f32.T) 报
    #   "expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float"
    # 质心探测是度量计算, 统一上浮到 float32 (bf16→f32 无精度损失), 标签转成 long 作索引 dtype。
    if hidden_states.dtype != torch.float32:
        hidden_states = hidden_states.float()
    labels = labels.long() if hasattr(labels, 'long') else torch.as_tensor(labels, device=hidden_states.device, dtype=torch.long)
    centroids = torch.zeros((num_classes, hidden_states.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    counts = torch.zeros(num_classes, device=hidden_states.device)
    
    for i in range(hidden_states.shape[0]):
        label = labels[i]
        centroids[label] += hidden_states[i]
        counts[label] += 1
    
    # 避免除零
    centroids = centroids / counts.unsqueeze(1).clamp(min=1)
    
    # 计算每个样本到质心的余弦相似度
    # dists: [N, num_classes]
    norms = torch.norm(hidden_states, dim=1, keepdim=True)
    c_norms = torch.norm(centroids, dim=1, keepdim=True).T
    sims = torch.mm(hidden_states, centroids.T) / (norms * c_norms + 1e-8)
    
    preds = torch.argmax(sims, dim=1)
    acc = (preds == labels).float().mean().item()
    return acc

def find_model_path(keyword):
    import os
    print(f"Searching for model containing '{keyword}' in /kaggle/input...")
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'config.json' in files:
            if keyword.lower() in root.lower():
                print(f"Found model at: {root}")
                return root
    return None

def run_experiment():
    # 1. 初始化
    print("Initializing Instacart Cognitive Space Engine...")
    # 这里的 CORPUS_B64 和 QUANT 会由 make_arms.py 替换
    corpus = load_embedded_corpus("#__CORPUS_B64__#")
    quant = "#__QUANT__#"
    
    if quant == "4bit":
        print("Installing bitsandbytes for 4bit quantization...")
        _osx("python -m pip install -q -U 'bitsandbytes>=0.46.1'")
    
    if not corpus:
        print("ERROR: Corpus not found!")
        return

    # 动态寻找模型路径
    model_keyword = "0.5b" if quant == "none" else "7b"
    model_id = find_model_path(model_keyword)
    
    if model_id:
        print(f"Using local mount: {model_id}")
    else:
        print(f"Local model not found, falling back to HF Hub for {model_keyword}...")
        model_id = "Qwen/Qwen2.5-0.5B-Instruct" if quant == "none" else "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"Loading model: {model_id} (quant={quant})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 量化配置
    model_kwargs = {"device_map": "auto", "trust_remote_code": True, "attn_implementation": "eager", "output_attentions": True}
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    # 2. 准备数据
    # 我们需要：
    # - Aisle 标注: {product_name: aisle_id}
    # - Dept 标注: {product_name: dept_id}
    # - 意图标注: {order_id: intent_id}
    
    all_products = []
    all_labels_aisle = []
    all_labels_dept = []
    
    # 建立唯一映射
    all_aisles = sorted(list(set([item['aisle'] for order in corpus for item in order['items']])))
    all_depts = sorted(list(set([item['dept'] for order in corpus for item in order['items']])))
    aisle_to_id = {name: i for i, name in enumerate(all_aisles)}
    dept_to_id = {name: i for i, name in enumerate(all_depts)}
    
    # 抽样产品进行层级探测 (Centroid Probing)
    sampled_orders = random.sample(corpus, min(len(corpus), PROBE_SAMPLES))
    for order in sampled_orders:
        for item in order['items']:
            all_products.append(item['product'])
            all_labels_aisle.append(aisle_to_id[item['aisle']])
            all_labels_dept.append(dept_to_id[item['dept']])
    
    # 限制样本量防止 OOM
    all_products = all_products[:MODEL_SAMPLES]
    all_labels_aisle = all_labels_aisle[:MODEL_SAMPLES]
    all_labels_dept = all_labels_dept[:MODEL_SAMPLES]
    
    # 3. 逐层提取隐藏状态
    print(f"Extracting hidden states for {len(all_products)} products...")
    layer_acc_aisle = []
    layer_acc_dept = []
    
    # 将产品分批处理
    batch_size = 8
    all_hidden_states = [] # [Layers, Samples, Dim]
    
    with torch.no_grad():
        for i in range(0, len(all_products), batch_size):
            batch = all_products[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            # 取最后一个 token 的 hidden states (last token representation)
            # outputs.hidden_states is tuple of (batch, seq, dim)
            batch_h = [h[:, -1, :].detach().cpu() for h in outputs.hidden_states]
            all_hidden_states.append(batch_h)

    # 汇总所有 batch: [Layers, Samples, Dim]
    # all_hidden_states 结构是 [batches][layers][batch_size, dim]
    num_layers = len(model.config.layers) if hasattr(model.config, 'layers') else len(model.model.layers)
    final_h = [] 
    for l in range(num_layers + 1): # include embedding layer
        layer_data = torch.cat([b[l] for b in all_hidden_states], dim=0)
        final_h.append(layer_data)

    # 4. 运行质心探测 (Centroid Probing)
    print("Running Centroid Probing across layers...")
    labels_aisle = torch.tensor(all_labels_aisle)
    labels_dept = torch.tensor(all_labels_dept)
    
    for l in range(num_layers + 1):
        h = final_h[l].to(model.device)
        acc_a = get_centroid_acc(h, labels_aisle.to(model.device), len(all_aisles))
        acc_d = get_centroid_acc(h, labels_dept.to(model.device), len(all_depts))
        layer_acc_aisle.append(acc_a)
        layer_acc_dept.append(acc_d)
        print(f"Layer {l:02d} | Aisle Acc: {acc_a:.4f} | Dept Acc: {acc_d:.4f}")

    # 5. 意图空间分析 (Intent Analysis)
    print("Analyzing Intent Space...")
    intent_labels = {"Healthy": 0, "Bulk": 1, "Convenience": 2, "General": 3}
    intent_centroids = torch.zeros((4, model.config.hidden_size), device=model.device)
    intent_counts = torch.zeros(4, device=model.device)
    
    # 抽样订单计算意图向量 (Mean Pool of products in order)
    sampled_intent_orders = random.sample(corpus, min(len(corpus), INTENT_SAMPLES))
    for order in sampled_intent_orders:
        # 订单内商品
        prods = [item['product'] for item in order['items']][:SAMPLES_PER_ORDER]
        inputs = tokenizer(prods, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 用最后一层 Mean Pool
            h = outputs.hidden_states[-1][:, -1, :].mean(dim=0)
            
            label = intent_labels.get(order['intent'], 3)
            intent_centroids[label] += h
            intent_counts[label] += 1
            
    intent_centroids = intent_centroids / intent_counts.unsqueeze(1).clamp(min=1)
    
    # 计算意图间距离
    dist_h_b = torch.norm(intent_centroids[0] - intent_centroids[1]).item()
    dist_g_b = torch.norm(intent_centroids[3] - intent_centroids[1]).item()
    print(f"Intent Dist: Healthy-Bulk {dist_h_b:.4f}, General-Bulk {dist_g_b:.4f}")

    # 6. 结果落盘
    metrics = {
        "arm": "0.5b" if quant == "none" else "7b",
        "model": {"mount": model_id, "quant": quant},
        "layer_acc_aisle": layer_acc_aisle,
        "layer_acc_dept": layer_acc_dept,
        "intent_dist": {"h_b": dist_h_b, "g_b": dist_g_b},
        "elapsed_s": time.time() - start_time if 'start_time' in globals() else 0
    }
    
    with open('metrics_instacart_kaggle.json', 'w') as f:
        json.dump(metrics, f, indent=1)
    
    # 哨兵信号
    print(f"REPORT_LINE::LayerAisleDone::{json.dumps(layer_acc_aisle)}")
    print(f"REPORT_LINE::LayerDeptDone::{json.dumps(layer_acc_dept)}")
    print(f"REPORT_LINE::IntentDistDone::{json.dumps(metrics['intent_dist'])}")
    
    # 最终快照
    print(f"INSTACART_DONE::{base64.b64encode(json.dumps(metrics).encode()).decode()}")

if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
