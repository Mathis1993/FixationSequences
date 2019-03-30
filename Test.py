#!python3
import pandas as pd

pd.read_json("allImages_unfolded.json", orient="split")
print("done")