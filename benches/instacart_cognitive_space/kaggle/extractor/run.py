import pandas as pd
import json
import random
import os

def extract_instacart_balanced(n_per_intent=100, output_file='corpus_instacart.json'):
    print("Loading Instacart data...")
    base_path = '/kaggle/input/instacart-market-basket-analysis/'
    
    products = pd.read_csv(base_path + 'products.csv')
    aisles = pd.read_csv(base_path + 'aisles.csv')
    departments = pd.read_csv(base_path + 'departments.csv')
    order_products = pd.read_csv(base_path + 'order_products__prior.csv')
    
    prod_meta = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')
    meta_map = prod_meta.set_index('product_id')[['product_name', 'aisle', 'department']].to_dict('index')
    
    # Strict Intent Definitions
    intent_rules = {
        "Healthy": {"aisles": ["fresh fruits", "fresh vegetables", "packaged vegetables fruits"], "keywords": ["organic"]},
        "Bulk": {"aisles": ["paper products", "household", "frozen"], "keywords": []},
        "Convenience": {"aisles": ["snacks", "beverages", "candy"], "keywords": []},
        "General": {"aisles": [], "keywords": []}
    }
    
    corpus = []
    sampled_order_ids = set()
    
    for intent, rules in intent_rules.items():
        if intent == "General": continue
        
        # Use a safe way to filter
        rules_aisles = rules["aisles"]
        target_pids = prod_meta[prod_meta['aisle'].isin(rules_aisles)]['product_id'].tolist()
        
        matching_orders = order_products[order_products['product_id'].isin(target_pids)]['order_id'].unique()
        
        sampled = random.sample(list(matching_orders), min(n_per_intent, len(matching_orders)))
        sampled_order_ids.update(sampled)
        
    all_unique_orders = order_products['order_id'].unique()
    remaining = list(set(all_unique_orders) - sampled_order_ids)
    general_sampled = random.sample(remaining, min(n_per_intent, len(remaining)))
    sampled_order_ids.update(general_sampled)
    
    for oid in sampled_order_ids:
        prods_in_order = order_products[order_products['order_id'] == oid]['product_id'].tolist()
        order_data = []
        for pid in prods_in_order:
            if pid in meta_map:
                order_data.append({
                    'product': meta_map[pid]['product_name'],
                    'aisle': meta_map[pid]['aisle'],
                    'dept': meta_map[pid]['department']
                })
        
        if order_data:
            all_names = " ".join([p['product'] for p in order_data]).lower()
            all_aisles = [p['aisle'] for p in order_data]
            
            if any("organic" in p['product'].lower() for p in order_data) or any(a in intent_rules["Healthy"]["aisles"] for a in all_aisles):
                intent = "Healthy"
            elif any(p['aisle'] in intent_rules["Bulk"]["aisles"] for p in order_data):
                intent = "Bulk"
            elif any(p['aisle'] in intent_rules["Convenience"]["aisles"] for p in order_data):
                intent = "Convenience"
            else:
                intent = "General"
                
            corpus.append({
                'order_id': int(oid),
                'intent': intent,
                'items': order_data
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=1)
    
    print(f"Successfully extracted balanced corpus: {len(corpus)} orders")

if __name__ == "__main__":
    extract_instacart_balanced()
