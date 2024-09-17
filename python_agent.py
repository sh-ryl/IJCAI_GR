import os
import random
import sys
import numpy as np
import decimal
import scenario
from random import randrange

from cooperative_craft_world import CooperativeCraftWorld

from neural_q_learner import NeuralQLearner
from dqn import DQN_Config

from goal_recogniser import GoalRecogniser

import torch

from copy import deepcopy

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


def goal_dic_to_str(goal_dic, inc_weight):
    # *S* why is this function outside of the class?
    goal_dic_keys = list(goal_dic.keys())
    goal_str = goal_dic_keys[0]
    if inc_weight:
        goal_str += "_" + str(goal_dic[goal_dic_keys[0]])
        for i in range(1, len(goal_dic_keys)):
            goal_str += "_" + goal_dic_keys[i] + \
                "_" + str(goal_dic[goal_dic_keys[i]])
    else:
        for i in range(1, len(goal_dic_keys)):
            goal_str += "_" + goal_dic_keys[i]

    return goal_str


def reset_all():

    global agent, total_reward, num_trials, seed, state, gr_obs, GR

    num_trials += 1

    # Only reset the environment seed when we're up to a new agent combination (to remove bias from the evaluation).
    if num_seeds != -1:
        seed = (seed + 1) % num_seeds
    else:
        seed = random.randrange(sys.maxsize)

    if agent_params["test_mode"]:
        print("\nSetting environment seed = " + str(seed) + "...")

    total_reward = 0

    agent.reset(
        goal_dic, current_scenario["externally_visible_goal_sets"][0], model_file)
    # will only reset with model file if it's on evaluation mode

    state = env.reset([agent], seed)

    if gr_obs:
        GR.reset()

    # Give starting items (if applicable).
    for i in range(len([agent])):
        for item, count in current_scenario["starting_items"][i].items():
            state.inventory[i][item] = count

    # Make the tasks easier at the beginning of training by gifting some starting items (gradually phased out).
    if not agent_params["test_mode"]:
        start_with_item_pr = 0.5 * max(1.0 - frame_num / 1000000, 0)
        for k, v in state.inventory[0].items():
            if np.random.uniform() < start_with_item_pr:
                state.inventory[0][k] += 1
# endregion


# region ENV INIT
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
goal_dic = goal_sets[0]

if "externally_visible_goal_sets" not in current_scenario:
    current_scenario["externally_visible_goal_sets"] = goal_sets

if "num_spawned" not in current_scenario:
    current_scenario["num_spawned"] = scenario.scenarios["default"]["num_spawned"]

if "regeneration" not in current_scenario:
    current_scenario["regeneration"] = scenario.scenarios["default"]["regeneration"]

if "starting_items" not in current_scenario:
    current_scenario["starting_items"] = scenario.scenarios["default"]["starting_items"]

if "hidden_items" not in current_scenario:
    current_scenario["hidden_items"] = scenario.scenarios["default"]["hidden_items"]

agent_params = {}

if sys.argv[1] == "train":
    agent_params["test_mode"] = False
    n_agents = 1
    if torch.cuda.is_available():
        gpu = 0
        print("Training using CUDA")
    else:
        gpu = -1
        print("Training using CPU")

else:
    print("Testing with CPU")
    agent_params["test_mode"] = True
    n_agents = 1  # 2 # Change for this code since we are only doing single agent GR
    gpu = -1  # Use CPU when not training

env_render = False
belief = False
limit = False
gr_obs = False
ability = False
uvfa = False

IO_param_path = ''
IO_param = sys.argv[2::]
ab_rating = {}

if "render" in IO_param:
    env_render = True
    IO_param.remove('render')
    print("Rendering environment ON")
else:
    print("Rendering environment OFF")

if "GR" in IO_param:
    gr_obs = True
    IO_param.remove('GR')
    print("GR Observer is ON")
else:
    print("GR Observer is OFF")

if "limit" in IO_param:
    limit = True
    print("Max inventory for collectible items (grass, iron, and wood) is LIMITED to 1")
    IO_param_path += 'limit/'
else:
    print("Max inventory is 999 for all ingredients")

if "uvfa" in IO_param:
    uvfa = True
    print("UVFA is ON")
    IO_param_path += 'uvfa/'
else:
    print("UVFA is OFF")

if "belief" in IO_param:
    belief = True
    print("Using hidden items")
    IO_param_path += 'belief/'

if "ability" in IO_param:
    ability = True
    print("Using skills")
    IO_param_path += 'skill/'
    ab_rating['player'] = 100
    ab_rating['craft'] = 100


env = CooperativeCraftWorld(current_scenario, size=size, n_agents=n_agents, allow_no_op=False, render=env_render,
                            ingredient_regen=current_scenario["regeneration"], max_steps=max_steps, IO_param=IO_param, ab_rating=ab_rating)
# endregion

# region PARAMETERS
agent_params["agent_type"] = "dqn"

# OPTIMIZER settings
agent_params["adam_lr"] = 0.000125
agent_params["adam_eps"] = 0.00015
agent_params["adam_beta1"] = 0.9
agent_params["adam_beta2"] = 0.999

# FILE I/O settings
real_path = os.path.dirname(os.path.realpath(__file__))

# Agent I/O settings
ag_models_folder = '/gr_model/' + IO_param_path

# saving/loading agent model for specific reward weightings
result_folder = ag_models_folder + goal_dic_to_str(goal_dic, inc_weight=True)
if uvfa:
    result_folder = ag_models_folder + goal_dic_to_str(goal_dic, inc_weight=False)
if belief:
    result_folder += "_hidden"
    for hi in current_scenario["hidden_items"][0]:
        result_folder += "_" + hi
if ability:
    result_folder += f"_level_{ab_rating['player']}_uniform"
    # uniform rating for all crafting skills
    # will look for ways to store skills with different difficulty ratings
agent_params["log_dir"] = real_path + f'{result_folder}/'

if not os.path.exists(agent_params["log_dir"]) and not gr_obs:
    os.makedirs(agent_params["log_dir"])
print(f"Agent model loaded: {result_folder}")

training_scores_file = "training_scores.csv"
with open(agent_params["log_dir"] + training_scores_file, 'a') as fd:
    fd.write('Frame,Score\n')

model_file = real_path + result_folder + \
    "/model.chk"  # To be used in eval mode only

if agent_params["test_mode"]:
    testing_scores_file = 'testing_scores.csv'
    with open(agent_params["log_dir"] + testing_scores_file, 'w') as fd:
        fd.write('seed,agent,agent_score\n')

agent_params["saved_model_dir"] = os.path.dirname(
    os.path.realpath(__file__)) + '/saved_models/'

# Observer (GR) I/O Settings
if gr_obs:
    gr_models_folder = '/ag_model/'
    model_dir = real_path + gr_models_folder
    print(f"GR model folder: {gr_models_folder}")
    goal_log_path = real_path + '/gr_log/' + \
        goal_dic_to_str(goal_dic, inc_weight=False)  # doesn't include weight
    if not os.path.exists(goal_log_path):
        os.makedirs(goal_log_path)

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
if uvfa:
    max_training_frames = 999999999
steps_since_eval_ran = 0
steps_since_eval_began = 0
eval_total_score = 0
eval_total_episodes = 0
best_eval_average = float("-inf")
episode_done = False
# endregion

# region AGENT INIT
# Initialise agent
agent = NeuralQLearner("Q_learner", agent_params, transition_params)
agent_combos = [[agent]]

reward = 0
total_reward = 0

num_trials = 0

# So that we reset seeds during the first call of reset_all().
seed = -1
state = None
# endregion

# region GR INIT

if gr_obs:
    gr_dqn_config = deepcopy(agent_params["dqn_config"]) # creates a separate config for gr and agent
    GR = GoalRecogniser(goal_list=list(goal_dic.keys()), saved_model_dir=model_dir,
                        dqn_config=gr_dqn_config, log_dir=goal_log_path, IO_param=IO_param)
# endregion

reset_all()

if gr_obs:
    GR.set_external_agent(agent)

# region MAIN LOOP
input("Start?")
while frame_num < max_training_frames:
    a = agent.perceive(reward, state, episode_done, eval_running)

    state, reward_list, episode_done, info = env.step_full_state(a)

    if gr_obs:
        GR.perceive(state, a)

    reward = reward_list[0]  # reward is list with length based on num_agents

    total_reward += reward

    if eval_running:
        steps_since_eval_began += 1
    else:
        steps_since_eval_ran += 1
        frame_num += 1

    if episode_done:
        if eval_running:  # This is only run during training
            print('Evaluation time step: ' + str(steps_since_eval_began) +
                  ', episode ended with score: ' + str(total_reward))
            eval_total_score += total_reward
            eval_total_episodes += 1
        else:
            score_str = ''

            average_total_reward = total_reward / num_trials

            if agent_params["test_mode"]:
                score_str = score_str + ', ' + agent.name + ": " + str(
                    total_reward) + " (" + "{:.2f}".format(average_total_reward) + ")"
            else:
                score_str = score_str + ', ' + \
                    agent.name + ": " + str(total_reward)

            if not gr_obs: # if gr is off then print as normal
                print('Time step: ' + str(frame_num) + ', ep scores:' + score_str[1:])

            if agent_params["test_mode"]:
                with open(agent_params["log_dir"] + testing_scores_file, 'a') as fd:
                    fd.write("'" + float_to_str(seed) + ',' +
                             agent.name + ',' + str(total_reward) + '\n')

        if gr_obs and not env_render: # pause before going to the next episode for gr obs
            print(f"Agent model loaded: {result_folder}")
            input()

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

                with open(agent_params["log_dir"] + training_scores_file, 'a') as fd:
                    fd.write(str(agent.numSteps) + ',' +
                             str(ave_eval_score) + '\n')

                if ave_eval_score > best_eval_average:
                    best_eval_average = ave_eval_score
                    print('New best eval average of ' +
                          str(best_eval_average))
                    agent.save_model()
                else:
                    print('Did not beat best eval average of ' +
                          str(best_eval_average))

                eval_running = False
                steps_since_eval_began = 0

    if gr_obs and not env_render:
        print('Time step: ' + str(frame_num) + ', total reward:' + str(total_reward))
        print()

    if env_render:
        print()
        print(f"Agent model loaded: {result_folder}")
        print(f"Goal sets: {goal_dic.items()}")
        print(f"Total reward: {total_reward}")
        print(f"Timestep: {frame_num}")
        input()

# endregion
