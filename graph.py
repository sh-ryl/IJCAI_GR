# Re-import necessary libraries for data handling and plotting
import pandas as pd
import matplotlib.pyplot as plt

# File paths of CSVs
# cloth_stick
# root = "mod/gr_log/cloth_stick"
# ag_wc = range(1, 10, 2)
# ag_weight_combo = [
#     f'/cloth_{round(x/10,1)}_stick_{round(1-x/10,1)}' for x in ag_wc]
# ag_titles = [
#     f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in ag_wc]
# gr_wc = range(1, 10, 2)
# gr_weight_combo = [
#     f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in gr_wc]
# dkl_y = [0.9, 2.5]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.84, 0.37)

# UVFA cloth_stick
root = "mod/gr_log/cloth_stick"
ag_wc = range(1, 10, 2)
ag_weight_combo = [
    f'/cloth_{round(x/10,1)}_stick_{round(1-x/10,1)}_uvfa' for x in ag_wc]
ag_titles = [
    f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in ag_wc]
gr_wc = range(1, 10, 1)
gr_weight_combo = [
    f"cloth: {round(x/10,1)}, stick: {round(1-x/10,1)}" for x in gr_wc]
dkl_y = [0.9, 2.5]
bi_y = [-0.05, 1.05]
legend_loc = (0.84, 0.37)

# axe_bridge
# root = "mod/gr_log/axe_bridge"
# ag_wc = [100, 500]
# ag_weight_combo = [f'/axe_1_bridge_0.7_ability_{lv}_incentive' for lv in ag_wc]
# ag_titles = [f"level: {lv}" for lv in ag_wc]
# gr_wc = ag_wc
# gr_weight_combo = [f"level: {lv}" for lv in ag_wc]
# dkl_y = [0.9, 2.5]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.37, 0.45)

# iron_wood_grass
# root = "mod/gr_log/iron_wood_grass"
# ag_wc = [(1, -1, ""), (-1, 1, ""), (1, -1, "_belief"), (-1, 1, "_belief")]
# ag_weight_combo = [f'/iron_0.7_wood_{w[0]}_grass_{w[1]}{w[2]}' for w in ag_wc]
# ag_titles = ["full obs with grass -1", "full obs with wood -1",
#              "part obs with grass -1", "part obs with wood -1"]
# gr_weight_combo = ["full obs with wood -1", "full obs with grass -1",
#                    "part obs with grass -1", "part obs with wood -1"]
# dkl_y = [0.5, 3.8]
# bi_y = [-0.05, 1.05]
# legend_loc = (0.68, 0.42)

# target
target_file_1 = "_avg_dkl_zbc.csv"
target_file_2 = "_avg_bi_prob.csv"

# Plotting 5 horizontal subplots
total_plot = 6
fig, axs = plt.subplots(2, total_plot, figsize=(15, 5))

# Load the datasets
ag_df_1 = [pd.read_csv(root+combo+target_file_1, header=None)
           for combo in ag_weight_combo]
ag_df_2 = [pd.read_csv(root+combo+target_file_2, header=None)
           for combo in ag_weight_combo]

# colors = ['blue', 'orange', 'green', 'red', 'purple']

# Plot each dataset on a separate subplot
for row in range(2):
    data = ag_df_1
    if row == 1:
        data = ag_df_2
    for ag in range(len(ag_wc)):
        for gr in range(len(gr_weight_combo)):
            y = data[ag].iloc[:, gr].values
            x = range(len(y))
            axs[row, ag].plot(
                x, y, label=gr_weight_combo[gr])

            if row == 0:
                axs[row, ag].set_title(ag_titles[ag], fontsize=12)
                axs[row, ag].set_ylim(dkl_y)
                if ag == 0:
                    axs[row, ag].set_ylabel("Running Average DKL")
            elif row == 1:
                axs[row, ag].set_ylim(bi_y)
                if ag == 0:
                    axs[row, ag].set_ylabel("Bayesian Inference Prob.")

            axs[row, ag].set_xlabel("timestep")

# Clear the last subplot
for x in range(total_plot-len(ag_wc)):
    axs[0, x+len(ag_wc)].axis("off")
    axs[1, x+len(ag_wc)].axis("off")

# Add a common legend outside all subplots
handles, labels = axs[1, ag].get_legend_handles_labels()
fig.legend(handles, labels, loc=legend_loc, title="GR Observer")

# Adjust layout
plt.tight_layout()
plt.show()
