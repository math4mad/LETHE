import os, sys, base64, gzip, json, shutil

MODEL_SLUGS = {
    "none": "qwen-lm/qwen2.5/transformers/0.5b-instruct/1",
    "4bit": "qwen-lm/qwen2.5/transformers/7b-instruct/1",
}

def generate_arm(arm_id, quant, corpus_path, template_path, out_dir):
    print(f"Generating arm {arm_id} (quant={quant})...")
    model_slug = MODEL_SLUGS.get(quant)
    if not model_slug:
        raise ValueError(f"Unknown quant {quant!r}; must be one of {list(MODEL_SLUGS)}")

    # 1. Load and compress corpus
    with open(corpus_path, 'rb') as f:
        raw_data = f.read()
    
    compressed = gzip.compress(raw_data)
    b64_corpus = base64.b64encode(compressed).decode('utf-8')
    
    # 2. Read template and replace
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("#__CORPUS_B64__#", b64_corpus)
    content = content.replace("#__QUANT__#", quant)
    
    # 3. Write to output dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'run.py'), 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 4. Write metadata
    meta = {
        # id 必须与 title-slug 一致(全小写连字符): title "Instacart Cognitive Space - 0-5b"
        # → slug "instacart-cognitive-space-0-5b"。否则 push 报 409 "title does not resolve to specified id"。
        "id": f"math4amd/instacart-cognitive-space-{arm_id}",
        "title": f"Instacart Cognitive Space - {arm_id}",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False if quant == "none" else True,
        # 关键修复: 声明 model_sources (挂载名) —— 之前缺此字段, /kaggle/input/models/... 未挂上,
        # from_pretrained 走 HF Hub fallback 报 "Repo id must be in the form..." (0-5b) / bnb 4bit 死局 (7b)
        "model_sources": [model_slug],
    }
    with open(os.path.join(out_dir, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    
    print(f"  model_sources: {model_slug}")
    
    print(f"Arm {arm_id} generated in {out_dir}")

if __name__ == "__main__":
    # config
    ROOT = "/Users/mac/Programming/code-2026/Concept-Space-Sphere/benches/instacart_cognitive_space/kaggle"
    TEMPLATE = os.path.join(ROOT, "run_template.py")
    CORPUS = os.path.join(ROOT, "corpus_instacart.json")
    
    if not os.path.exists(CORPUS):
        print(f"ERROR: Corpus file {CORPUS} not found! Please run extract_corpus.py on Kaggle and download first.")
        sys.exit(1)
        
    # Arm A: 0.5B bf16
    generate_arm("0-5b", "none", CORPUS, TEMPLATE, os.path.join(ROOT, "instacart-0-5b"))
    # Arm B: 7B 4bit
    generate_arm("7b", "4bit", CORPUS, TEMPLATE, os.path.join(ROOT, "instacart-7b"))
