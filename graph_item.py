import pandas as pd
import matplotlib.pyplot as plt
from cooperative_craft_world import _rewardable_items

# mod = "iron_wood_grass"
# wc = [(1, -1, ""), (-1, 1, ""), (1, -1, "_belief"), (-1, 1, "_belief")]
# weight_combo = [f'/iron_0.7_wood_{w[0]}_grass_{w[1]}{w[2]}' for w in wc]

# mod = "cloth_stick"
# wc = range(1, 10, 2)
# weight_combo = [
#     f'/cloth_{round(w/10,1)}_stick_{round(1-w/10,1)}' for w in wc]

mod = "cloth_stick"
wc = range(1, 10, 2)
weight_combo = [
    f'/cloth_{round(w/10,1)}_stick_{round(1-w/10,1)}_uvfa' for w in wc]

# mod = "axe_bridge"
# wc = [100, 500]
# weight_combo = [f'/axe_1_bridge_0.7_ability_{w}_incentive' for w in wc]

colors = ['blue', 'lightcoral', 'green', 'red', 'purple', 'brown',
          "orchid", "gray", "olive", "cyan", 'gold', 'lightseagreen']

# read file
fig, axs = plt.subplots(1, len(wc)+1, figsize=(15, 5), sharey=True)

# Load the datasets
item_df = [pd.read_csv(f"mod/gr_log/{mod}/{combo}_avg_item_count.csv", header=None)
           for combo in weight_combo]

for idf in range(len(wc)):
    for i in range(len(_rewardable_items)):
        y = item_df[idf].iloc[:, i].values
        x = range(len(y))
        axs[idf].plot(x, y, label=_rewardable_items[i], color=colors[i])

# Clear the last subplot
axs[idf+1].axis("off")

# Add a common legend outside all subplots
handles, labels = axs[idf].get_legend_handles_labels()
axs[idf+1].legend(handles, labels, loc='right', title="Inventory Items")
plt.show()
