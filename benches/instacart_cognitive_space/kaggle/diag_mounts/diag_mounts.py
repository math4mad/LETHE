import os
print("Kaggle Input Dir:")
try:
    print(os.listdir('/kaggle/input'))
except Exception as e:
    print(f"Error: {e}")
