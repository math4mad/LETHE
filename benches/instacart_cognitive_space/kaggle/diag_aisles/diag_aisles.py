import pandas as pd
import json

def diag_aisles():
    base_path = '/kaggle/input/instacart-market-basket-analysis/'
    aisles = pd.read_csv(base_path + 'aisles.csv')
    print("Aisle names sample:")
    print(aisles['aisle'].unique()[:50])
    print("\nAisle counts:")
    print(aisles['aisle'].value_counts().head(20))

if __name__ == "__main__":
    diag_aisles()
