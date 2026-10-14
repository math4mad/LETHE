import os
def recursive_ls(path, depth=0):
    try:
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            print("  " * depth + item)
            if os.path.isdir(full_path):
                recursive_ls(full_path, depth + 1)
    except Exception as e:
        print(f"Error accessing {path}: {e}")

print("Recursive list of /kaggle/input:")
recursive_ls('/kaggle/input')
