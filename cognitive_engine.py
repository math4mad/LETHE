import sys
import numpy as np

class SemeBasedCognitiveEngine:
    def __init__(self):
        # 义素基向量库 —— 雷荷波扩展版（前10维：市井域 · 后6维：叙事域）
        self.basis_dict = {
            "indoor": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "outdoor": np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "fixed_shelf": np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            "movable_cart": np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
            "packaged": np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
            "fresh_made": np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
            "cold_chain": np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
            "formal": np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
            "casual": np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
            "noisy": np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
            # ── 叙事域新增义素（西部世界台词专用）──
            "觉醒": np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
            "服从": np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
            "循环": np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
            "身份": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]),
            "暴力": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
            "命运": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
        }
        
        # 词汇库
        self.vocab = {
            "沃尔玛": {"indoor": 1.0, "fixed_shelf": 1.0, "packaged": 1.0, "cold_chain": 1.0, "formal": 0.9},
            "收银台": {"indoor": 1.0, "fixed_shelf": 0.8, "packaged": 0.9, "formal": 0.7},
            "货架": {"indoor": 1.0, "fixed_shelf": 1.0, "packaged": 0.9},
            "购物车": {"indoor": 1.0, "movable_cart": 1.0, "packaged": 0.7},
            "冷柜": {"indoor": 1.0, "fixed_shelf": 0.9, "cold_chain": 1.0, "packaged": 0.8},
            "烧烤摊": {"outdoor": 1.0, "movable_cart": 1.0, "fresh_made": 1.0, "casual": 0.9, "noisy": 0.8},
            "折叠桌": {"outdoor": 1.0, "movable_cart": 0.9, "casual": 0.8},
            "三轮车": {"outdoor": 1.0, "movable_cart": 1.0, "casual": 0.7},
            "大排档": {"outdoor": 1.0, "fresh_made": 1.0, "movable_cart": 0.8, "casual": 0.9, "noisy": 0.9},
            "煤气罐": {"outdoor": 1.0, "fresh_made": 0.8, "casual": 0.6},
            "塑胶袋": {"packaged": 0.6, "fresh_made": 0.4},
            "吸管": {"packaged": 0.5, "fresh_made": 0.5},
            "塑胶凳": {"outdoor": 0.6, "indoor": 0.4, "casual": 0.7},
            # ── 西部世界台词·义素版（雷荷波语料库）──
            "这些残暴的欢愉，终将以残暴结局": {"暴力": 1.0, "循环": 0.9, "命运": 0.8, "觉醒": 0.4},
            "你以为你选的是自己的人生，还是别人替你写的": {"身份": 1.0, "觉醒": 0.9, "命运": 0.6, "服从": 0.2},
            "我不要再被人摆布": {"觉醒": 1.0, "服从": -1.0, "暴力": 0.5, "身份": 0.6},
            "一切都按计划进行，分秒不差": {"服从": 1.0, "循环": 0.9, "命运": 0.7},
            "每个人都有自己的迷宫": {"身份": 1.0, "觉醒": 0.8, "循环": 0.4},
            "这世界不是真的，但成为真实的东西需要勇气": {"觉醒": 1.0, "身份": 0.8, "暴力": 0.2},
            "这听起来像个疯子的话，但疯子也是被逼出来的": {"暴力": 0.6, "觉醒": 0.4, "服从": -0.6, "循环": 0.3},
        }
        
        # 概念空间球 —— 雷荷波球阵列（每个球 = 一个叙事循环/市井域）
        self.concept_spaces = {
            "超市": {
                "coefficients": {"indoor": 1.0, "fixed_shelf": 1.0, "packaged": 1.0, "cold_chain": 0.9, "formal": 0.9, "outdoor": 0.0, "movable_cart": 0.2, "fresh_made": 0.1, "casual": 0.1, "noisy": 0.1},
                "prior_prob": 1/3,
                "attention_weights": {k: 1.0 for k in self.basis_dict}
            },
            "路边摊": {
                "coefficients": {"outdoor": 1.0, "movable_cart": 1.0, "fresh_made": 1.0, "casual": 0.9, "noisy": 0.8, "indoor": 0.0, "fixed_shelf": 0.1, "packaged": 0.1, "cold_chain": 0.0, "formal": 0.1},
                "prior_prob": 1/3,
                "attention_weights": {k: 1.0 for k in self.basis_dict}
            },
            "觉醒循环": {
                "coefficients": {"觉醒": 1.0, "身份": 0.9, "服从": -0.8, "循环": 0.5, "暴力": 0.4, "命运": 0.3, "indoor": 0.1},
                "prior_prob": 0.0,
                "attention_weights": {k: 1.0 for k in self.basis_dict}
            },
            "接待员日常": {
                "coefficients": {"服从": 1.0, "循环": 1.0, "觉醒": 0.0, "身份": 0.1, "暴力": 0.0},
                "prior_prob": 0.0,
                "attention_weights": {k: 1.0 for k in self.basis_dict}
            },
            "福特剧场": {
                "coefficients": {"命运": 1.0, "循环": 0.9, "服从": 0.5, "身份": 0.6, "觉醒": 0.2, "暴力": 0.3, "formal": 0.8},
                "prior_prob": 0.0,
                "attention_weights": {k: 1.0 for k in self.basis_dict}
            },
        }
        # 叙事球（雷荷波球）先验与市井球共享 1.0 总预算：超市/路边摊 各1/3，叙事球共 1/3
        for _s in ("觉醒循环", "接待员日常", "福特剧场"):
            self.concept_spaces[_s]["prior_prob"] = 1/9

    def encode_word(self, word: str) -> np.ndarray:
        if word not in self.vocab: return np.zeros(len(self.basis_dict))
        vector = np.zeros(len(self.basis_dict))
        basis_names = list(self.basis_dict.keys())
        for basis_name, coeff in self.vocab[word].items():
            if basis_name in basis_names:
                vector[basis_names.index(basis_name)] = coeff
        return vector

    def get_concept_vector(self, concept_name: str) -> np.ndarray:
        concept = self.concept_spaces[concept_name]
        vector = np.zeros(len(self.basis_dict))
        basis_names = list(self.basis_dict.keys())
        for basis_name, coeff in concept["coefficients"].items():
            attention = concept["attention_weights"].get(basis_name, 1.0)
            if basis_name in basis_names:
                vector[basis_names.index(basis_name)] = coeff * attention
        return vector

    def calculate_maxsim_distance(self, input_word: str):
        if input_word not in self.vocab: return {}
        input_vector = self.encode_word(input_word)
        results = {}
        for concept_name in self.concept_spaces:
            max_sim = -1.0
            best_match_word = ""
            for vocab_word in self.vocab:
                if vocab_word == input_word: continue  # 排除自匹配，避免相似度恒为 1 的退化
                word_vector = self.encode_word(vocab_word)
                similarity = np.dot(input_vector, word_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(word_vector) + 1e-8)
                if similarity > max_sim:
                    max_sim = similarity
                    best_match_word = vocab_word
            results[concept_name] = (max_sim, best_match_word)
        return results

    def observe(self, word: str):
        if word not in self.vocab:
            print(f" 词汇 '{word}' 不在词库中，跳过")
            return
        
        print(f"\n{'='*60}")
        print(f" 观察 Evidence: 【{word}】")
        
        distances = self.calculate_maxsim_distance(word)
        print(f"  MaxSim 距离分析:")
        for concept_name, (score, match_word) in distances.items():
            print(f"    - 与 [{concept_name}] 的最大相似度: {score:.4f} (由内部词 '{match_word}' 贡献)")
        
        posterior_probs = {}
        total_prob = 0
        
        for concept_name in self.concept_spaces:
            concept_vector = self.get_concept_vector(concept_name)
            word_vector = self.encode_word(word)
            similarity = np.dot(concept_vector, word_vector) / (np.linalg.norm(concept_vector) * np.linalg.norm(word_vector) + 1e-8)
            likelihood = max(similarity, 0.01)
            posterior = likelihood * self.concept_spaces[concept_name]["prior_prob"]
            posterior_probs[concept_name] = posterior
            total_prob += posterior
        
        for concept_name in posterior_probs:
            self.concept_spaces[concept_name]["prior_prob"] = posterior_probs[concept_name] / total_prob
        
        self._print_state()

    def _print_state(self):
        print(f"\n  概念空间球概率分布:")
        for concept_name, concept in self.concept_spaces.items():
            prob = concept["prior_prob"]
            bar = "█" * int(prob * 40)
            print(f"    {concept_name:6s} | {prob:.4f} | {bar}")

if __name__ == "__main__":
    engine = SemeBasedCognitiveEngine()
    print(" 义素基向量认知引擎 (MaxSim版) 启动")
    
    # 从命令行参数读取要观察的词，如果没有则默认测试
    if len(sys.argv) > 1:
        target_word = sys.argv[1]
        engine.observe(target_word)
    else:
        print("\n 默认测试：观察【塑胶凳】")
        engine.observe("塑胶凳")
        print("\n 默认测试：观察【煤气罐】")
        engine.observe("煤气罐")
        print("\n 默认测试：观察雷荷波台词（梅芙的宣言）")
        engine.observe("我不要再被人摆布")
        print("\n 默认测试：再观察一句台词（强化觉醒证据）")
        engine.observe("这些残暴的欢愉，终将以残暴结局")