import os
import random
import sys
import numpy as np
import decimal
import scenario
from cooperative_craft_world import CooperativeCraftWorld
from random import randrange
from neural_q_learner import NeuralQLearner
from dqn import DQN_Config

ctx = decimal.Context()
ctx.prec = 20

# region HELPER FUNCTIONS
def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def reset_all():

    global agents, agent_combo_idx, total_reward, num_trials, seed, state

    agent_combo_idx += 1
    if agent_combo_idx >= len(agent_combos):
        agent_combo_idx = 0
        num_trials += 1

        # Only reset the environment seed when we're up to a new agent combination (to remove bias from the evaluation).
        if num_seeds != -1:
            seed = (seed + 1) % num_seeds
        else:
            seed = random.randrange(sys.maxsize)

        if agent_params["test_mode"]:
            print("\nSetting environment seed = " + str(seed) + "...")

    agents = agent_combos[agent_combo_idx]

    total_reward = np.zeros((n_agents), dtype=np.float32)

    for i in range(len(agents)):
        agents[i].reset(i, seed, goal_sets[i],
                        current_scenario["externally_visible_goal_sets"][i], model_file)
                        # will only reset with model file if it's on evaluation mode

    state = env.reset(agents, seed)

    # Give starting items (if applicable).
    for i in range(len(agents)):
        for item, count in current_scenario["starting_items"][i].items():
            state.inventory[i][item] = count

    # Make the tasks easier at the beginning of training by gifting some starting items (gradually phased out).
    if not agent_params["test_mode"]:
        start_with_item_pr = 0.5 * max(1.0 - frame_num / 1000000, 0)
        for k, v in state.inventory[0].items():
            if np.random.uniform() < start_with_item_pr:
                state.inventory[0][k] += 1
# endregion

# region ENV SETUP
if len(sys.argv) < 2:
    print('Usage:', sys.argv[0], 'scenario')
    sys.exit()

if sys.argv[1] not in scenario.scenarios:
    print("Unknown scenario: " + sys.argv[1])
    sys.exit()

env_name = "cooperative_craft_world"
num_seeds = -1
max_steps = 100
size = (7, 7)

current_scenario = scenario.scenarios[sys.argv[1]]

goal_sets = current_scenario["goal_sets"]

if "externally_visible_goal_sets" not in current_scenario:
    current_scenario["externally_visible_goal_sets"] = goal_sets

if "num_spawned" not in current_scenario:
    current_scenario["num_spawned"] = scenario.scenarios["default"]["num_spawned"]

if "regeneration" not in current_scenario:
    current_scenario["regeneration"] = scenario.scenarios["default"]["regeneration"]

if "starting_items" not in current_scenario:
    current_scenario["starting_items"] = scenario.scenarios["default"]["starting_items"]

agent_params = {}

if sys.argv[1] == "train":
    agent_params["test_mode"] = False
    n_agents = 1
    gpu = -1
else:
    agent_params["test_mode"] = True
    n_agents = 1  # 2 # Change for this code since we are only doing single agent GR
    gpu = -1  # Use CPU when not training

if len(sys.argv) > 2:
    if sys.argv[2] == "render":
        env_render = True
else:
    env_render = False

env = CooperativeCraftWorld(current_scenario, size=size, n_agents=n_agents, allow_no_op=False,
                            render=env_render, ingredient_regen=current_scenario["regeneration"], max_steps=max_steps)
# endregion

# region PARAMETERS
agent_params["agent_type"] = "dqn"

# OPTIMIZER settings
agent_params["adam_lr"] = 0.000125
agent_params["adam_eps"] = 0.00015
agent_params["adam_beta1"] = 0.9
agent_params["adam_beta2"] = 0.999

# FILE I/O settings
goal_set_keys = list(goal_sets[0].keys())
result_folder = '/new_models/' + goal_set_keys[0] + "_" + str(goal_sets[0][goal_set_keys[0]])
for i in range(1, len(goal_set_keys)):
    result_folder = result_folder + "_" + \
        goal_set_keys[i] + "_" + str(goal_sets[0][goal_set_keys[i]])

agent_params["log_dir"] = os.path.dirname(os.path.realpath(__file__))
agent_params["log_dir"] = agent_params["log_dir"] + f'{result_folder}/'
if not os.path.exists(agent_params["log_dir"]):
    os.makedirs(agent_params["log_dir"])

training_scores_file = "training_scores.csv"
with open(agent_params["log_dir"] + training_scores_file, 'a') as fd:
    fd.write('Frame,Score\n')

model_file = result_folder + "/model.chk" # To be used in eval mode only 

if agent_params["test_mode"]:
    testing_scores_file = 'testing_scores.csv'
    with open(agent_params["log_dir"] + testing_scores_file, 'w') as fd:
        fd.write('seed,agent,agent_score\n')

# Set to None to disable logging
goal_recogniser_log_dir = agent_params["log_dir"]

agent_params["saved_model_dir"] = os.path.dirname(
    os.path.realpath(__file__)) + '/saved_models/'

# DQN Settings
agent_params["dqn_config"] = DQN_Config(
    env.observation_space.shape[0], env.action_space.n, gpu=gpu, noisy_nets=False, n_latent=64)

agent_params["n_step_n"] = 1
agent_params["max_reward"] = 2.0  # 1.0 # Use float("inf") for no clipping
agent_params["min_reward"] = -2.0  # -1.0 # Use float("-inf") for no clipping
agent_params["exploration_style"] = "e_greedy"  # e_greedy, e_softmax
agent_params["softmax_temperature"] = 0.05
agent_params["ep_start"] = 1
agent_params["ep_end"] = 0.01
agent_params["ep_endt"] = 1000000
agent_params["discount"] = 0.99

# To help with learning from very sparse rewards initially
agent_params["mixed_monte_carlo_proportion_start"] = 0.2
agent_params["mixed_monte_carlo_proportion_endt"] = 1000000

if agent_params["test_mode"]:
    agent_params["learn_start"] = -1  # Indicates no training
else:
    agent_params["learn_start"] = 50000

agent_params["update_freq"] = 4
agent_params["n_replay"] = 1
agent_params["minibatch_size"] = 32
agent_params["target_refresh_steps"] = 10000
agent_params["show_graphs"] = False
agent_params["graph_save_freq"] = 25000

# For training methods that require n step returns, set the below to True.
agent_params["post_episode_return_calcs_needed"] = True

# can be put together in evaluation setting since transition params is not using it
agent_params["eval_ep"] = 0.01

transition_params = {}
transition_params["agent_params"] = agent_params
transition_params["replay_size"] = 1000000
transition_params["bufferSize"] = 512
# endregion

# region EVAL SETTINGS
eval_freq = 250000  # As per Rainbow paper
eval_steps = 125000  # As per Rainbow paper
# Don't start evaluating until gifted items at the start of training are phased out.
eval_start_time = 1000000

eval_running = False  # different than test_mode in agent config
frame_num = 0
max_training_frames = 10000000  # 999999999
steps_since_eval_ran = 0
steps_since_eval_began = 0
eval_total_score = 0
eval_total_episodes = 0
best_eval_average = float("-inf")
episode_done = False
# endregion

# region AGENT INIT
# Initialise agent
agent_q_learner = NeuralQLearner("Q_learner", agent_params, transition_params)
agent_combos = [[agent_q_learner]]

reward = np.zeros((n_agents), dtype=np.float32)
total_reward = np.zeros((n_agents), dtype=np.float32)

sum_total_reward = np.zeros((len(agent_combos), n_agents), dtype=np.float32)
num_trials = 0

# So that we reset seeds during the first call of reset_all().
agent_combo_idx = len(agent_combos)
seed = -1
state = None
# endregion

# region MAIN LOOP
reset_all()

while frame_num < max_training_frames:
    agent_idx = state.player_turn

    a = agents[agent_idx].perceive(
        reward[agent_idx], state, episode_done, eval_running)

    state, reward, episode_done, info = env.step_full_state(a)

    for i in range(n_agents):
        total_reward[i] += reward[i]

    if eval_running:
        steps_since_eval_began += 1
    else:
        steps_since_eval_ran += 1
        frame_num += 1

    if episode_done:
        if eval_running:  # This is only run during training
            print('Evaluation time step: ' + str(steps_since_eval_began) +
                  ', episode ended with score: ' + str(total_reward[0]))
            eval_total_score += total_reward[0]
            eval_total_episodes += 1
        else:
            score_str = ''
            for i in range(0, n_agents):

                sum_total_reward[agent_combo_idx, i] += total_reward[i]
                average_total_reward = sum_total_reward[agent_combo_idx,
                                                        i] / num_trials

                if agent_params["test_mode"]:
                    score_str = score_str + ', ' + agents[i].name + ": " + str(
                        total_reward[i]) + " (" + "{:.2f}".format(average_total_reward) + ")"
                else:
                    score_str = score_str + ', ' + \
                        agents[i].name + ": " + str(total_reward[i])

            print('Time step: ' + str(frame_num) +
                  ', ep scores:' + score_str[1:])

            if agent_params["test_mode"]:
                with open(agent_params["log_dir"] + testing_scores_file, 'a') as fd:
                    fd.write("'" + float_to_str(seed) + ',' +
                             agents[0].name + ',' + str(total_reward[0]) + '\n')

        reset_all()

        # Model evaluation (only during Training)
        if not agent_params["test_mode"]:

            if frame_num >= eval_start_time and steps_since_eval_ran >= eval_freq:
                eval_running = True
                eval_total_score = 0
                eval_total_episodes = 0

                while steps_since_eval_ran >= eval_freq:
                    steps_since_eval_ran -= eval_freq

            elif steps_since_eval_began >= eval_steps:

                ave_eval_score = float(eval_total_score) / eval_total_episodes
                print('Evaluation ended with average score of ' +
                      str(ave_eval_score))

                if isinstance(agents[0], NeuralQLearner):
                    with open(agent_params["log_dir"] + training_scores_file, 'a') as fd:
                        fd.write(str(agents[0].numSteps) +
                                 ',' + str(ave_eval_score) + '\n')

                    if ave_eval_score > best_eval_average:
                        best_eval_average = ave_eval_score
                        print('New best eval average of ' +
                              str(best_eval_average))
                        agents[0].save_model()
                    else:
                        print('Did not beat best eval average of ' +
                              str(best_eval_average))

                eval_running = False
                steps_since_eval_began = 0

    if env_render:
        input()
        print("Goal sets", goal_sets[0].items())
        print(total_reward)
# endregion