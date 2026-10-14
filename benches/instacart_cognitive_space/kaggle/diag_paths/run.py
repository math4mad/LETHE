import os
def list_dir(path):
    print(f"Listing {path}...")
    try:
        for item in os.listdir(path):
            print(f"  {item}")
    except Exception as e:
        print(f"  Error: {e}")

# Datasets we intend to use
datasets = [
    '/kaggle/input/qwen2-5-transformers-0-5b-v1-tar',
    '/kaggle/input/qwen2-5-7b-instruct'
]

for d in datasets:
    list_dir(d)
