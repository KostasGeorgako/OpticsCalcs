import pandas as pd
import numpy as np

cart_df = pd.read_csv("../data/cart.csv")
total = sum(cart_df["total"])
print(f"Total cart cost: {total}")


