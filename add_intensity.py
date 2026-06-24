import pandas as pd

df = pd.read_csv("data/interim/balanced_dataset.csv")

def frp_to_intensity(frp):
    if frp < 5:
        return 1
    elif frp < 20:
        return 2
    elif frp < 50:
        return 3
    elif frp < 100:
        return 4
    else:
        return 5

df["fire_intensity"] = df["frp"].apply(frp_to_intensity)

df.to_csv("data/interim/balanced_dataset_multitask.csv", index=False)

print("New dataset created with intensity column")