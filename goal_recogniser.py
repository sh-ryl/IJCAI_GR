import torch
import torch.nn.functional as F
import string
import numpy as np
import agent
from cooperative_craft_world import CooperativeCraftWorldState
from dqn import DQN, DQN_Config
from dialog import Dialog

import os
import re
from copy import deepcopy

from itertools import chain

import pandas as pd
import matplotlib.pyplot as plt


class GoalRecogniser(object):

    def __init__(self, goal_list=[], model_temperature=0.1, hypothesis_momentum=0.9999, kl_tolerance=0.0, saved_model_dir=None, dqn_config: DQN_Config = None, show_graph=False, log_dir=None, IO_param=[], max_steps=100):

        self.saved_model_dir = saved_model_dir
        self.model_temperature = model_temperature
        self.hypothesis_momentum = hypothesis_momentum
        self.kl_tolerance = kl_tolerance
        self.show_graph = show_graph
        self.log_dir = log_dir
        self.first_log_write = True
        self.step_number = 0
        self.max_steps = max_steps

        self.device = torch.device("cuda" if dqn_config.gpu >= 0 else "cpu")
        self.probability_plot = Dialog()

        self.goal_list = goal_list

        # self.IO_param = IO_param

        self.trained_model_paths, self.trained_model_goal_dics, self.trained_model_param = self.find_folders(
            self.saved_model_dir, self.goal_list)

        self.trained_models = []
        self.tm_scores = [0] * len(self.trained_model_paths)
        self.tm_total_scores_each_step = [[0 for y in range(
            len(self.trained_model_paths))]for x in range(max_steps)]  # 100 for max timestep
        self.tm_scores_each_step = deepcopy(self.tm_total_scores_each_step)
        self.tm_state_size = [0] * len(self.trained_model_paths)

        for i in range(len(self.trained_model_paths)):
            model_file = f'{self.trained_model_paths[i]}/model.chk'
            checkpoint = torch.load(model_file, map_location=self.device)

            tm_state_size = checkpoint['model_state_dict']['fc1.weight'].size()[
                1]
            # deepcopy(dqn_config) # perhaps not using deepcopy would save mems, idk :p
            tm_dqn_config = dqn_config
            tm_dqn_config.set_state_size(tm_state_size)

            self.trained_models.append(DQN(tm_dqn_config))

            self.trained_models[i].load_state_dict(
                checkpoint['model_state_dict'])

            self.trained_model_paths[i] = self.trained_model_paths[i].removeprefix(
                self.saved_model_dir)

            # str.r
            print(
                f"GR model {i}: {self.trained_model_paths[i]}, Param: {self.trained_model_param[i]}")

    def set_external_agent(self, other_agent: agent.Agent):

        self.other_agent = other_agent

        self.probability_plot.reset()
        self.total_kl = np.zeros(
            (len(self.other_agent.externally_visible_goal_sets)), dtype=np.float32)
        self.total_kl_moving_avg = np.zeros(
            (len(self.other_agent.externally_visible_goal_sets)), dtype=np.float32)
        self.total_kl_moving_avg_debiased = np.zeros(
            (len(self.other_agent.externally_visible_goal_sets)), dtype=np.float32)
        self.moving_avg_updates = 0
        self.current_hypothesis = None
        self.step_number = 0

    def calculate_kl_divergence(self, model_no, state, observed_action: int, max_value=100.0):
        state = torch.from_numpy(state.getRepresentation(gr_obs=True,
                                                         gr_param=self.trained_model_param[model_no])).float().to(
            self.device).unsqueeze(0)
        q = self.trained_models[model_no].forward(
            state).cpu().detach().squeeze()
        probs = F.softmax(q.div(self.model_temperature), dim=0)
        return min(probs[observed_action].pow(-1).log().item(), max_value), probs

    def softmax(self, x, temperature):

        x = np.divide(x, temperature)
        # Necessary to ensure that the sum of the values is close enough to 1.
        e_x = np.exp(x - np.max(x)).astype(np.float64)
        result = e_x / e_x.sum(axis=0)
        result = result / result.sum(axis=0, keepdims=1)
        return result

    def perceive(self, state, action: int, frame_num, print_result):
        if print_result:
            print("GR observer")
            print(
                f"No.\t{'Score':<10} {'Parameters': <35} {'Goal set': <40} Action Probabilities")
        for i in range(len(self.trained_models)):
            kl_div, act_probs = self.calculate_kl_divergence(i, state, action)
            self.tm_scores[i] += kl_div
            self.tm_total_scores_each_step[frame_num %
                                           self.max_steps][i] += self.tm_scores[i]
            self.tm_scores_each_step[frame_num %
                                     self.max_steps][i] += kl_div
            act_probs = [round(x, 3) for x in act_probs.tolist()]
            if print_result:
                print(
                    f"{i}\t{round(self.tm_scores[i], 3):<10} {str(self.trained_model_param[i]): <35} {str(self.trained_model_goal_dics[i]): <40} {str(act_probs)}")

    def update_hypothesis(self):

        arg_min = np.argmin(self.total_kl_moving_avg_debiased)
        min_kl = self.total_kl_moving_avg_debiased[arg_min]

        selected_goals = []
        for i in range(0, len(self.total_kl_moving_avg_debiased)):
            if self.total_kl_moving_avg_debiased[i] <= min_kl + self.kl_tolerance:
                selected_goals.append(
                    self.other_agent.externally_visible_goal_sets[i])

        self.current_hypothesis = selected_goals[0]
        for i in range(1, len(selected_goals)):
            self.current_hypothesis = self.current_hypothesis + \
                "_and_" + selected_goals[i]

    def find_folders(self, directory, required_objects):
        trained_models_path = []
        trained_models_goal_dic = []
        trained_models_param = []
        param_set = {"limit", "belief", "uvfa", "ability"}

        # Iterate through the directory
        for root, dirs, files in os.walk(directory):
            # Get the directory name before the subdirectory
            # to split "\" because windows don't follow unix style
            base_folder = list(chain.from_iterable(
                [x.split("\\") for x in root.split('/')]))

            for folder_name in dirs:
                # filter out other folders
                if "uvfa" in base_folder:  # and "uvfa" in self.IO_param: # CHECK FOR UVFA LATER
                    pattern = r'([a-zA-Z]+_[a-zA-Z]+)'
                else:
                    # currently regex works if number is less than 1 with comma or more than 1 without comma
                    pattern = r'([a-zA-Z]+_-*\d+\.?\d*)'
                matches = [x.split('_')
                           for x in re.findall(pattern, folder_name)]
                if "uvfa" in base_folder:  # and "uvfa" in self.IO_param:
                    obj_list = matches[0]
                else:
                    obj_list = [x[0] for x in matches]
                    weight_list = [x[1] for x in matches]

                if "level" in obj_list:
                    obj_list.remove("level")

                if set(obj_list) == set(required_objects):
                    tm_param = {}
                    tm_goal_dic = {}

                    for param in param_set.intersection(base_folder):
                        tm_param[param] = ''

                    if "uvfa" in tm_param:
                        tm_param["uvfa"] = obj_list
                    else:

                        # get goal dic
                        for obj in required_objects:
                            tm_goal_dic[obj] = weight_list[obj_list.index(
                                obj)]

                        # get param
                        if "belief" in tm_param:
                            pattern = r'_hidden_([a-zA-Z]*)_([a-zA-Z]*)'
                            result = re.findall(pattern, folder_name)[0]
                            tm_param['belief'] = result

                    if "ability" in tm_param:
                        pattern = r'_level_(\d+)_uniform'
                        result = re.findall(pattern, folder_name)
                        tm_param['ability'] = result

                    trained_models_path.append(root+'/'+folder_name)
                    trained_models_param.append(tm_param)
                    trained_models_goal_dic.append(tm_goal_dic)

        combined = zip(trained_models_goal_dic,
                       trained_models_path, trained_models_param)
        sorted_combined = sorted(
            combined, key=lambda x: x[0][required_objects[0]])

        # Step 2: Extract sorted `d` and `other_list`
        trained_models_goal_dic, trained_models_path, trained_models_param = zip(
            *sorted_combined)

        return list(trained_models_path), list(trained_models_goal_dic), list(trained_models_param)

    def reset(self):
        self.tm_scores = [0] * len(self.trained_model_paths)

    def get_inference(self):
        temp_min = max(self.tm_scores)
        temp_min_id = 0

        for i in range(len(self.tm_scores)):
            if self.tm_scores[i] < temp_min:
                temp_min = self.tm_scores[i]
                temp_min_id = i

        return temp_min_id

    def get_result(self, max_frame, goal_str, ag_param=dict()):
        avg_total = []
        avg_step = []

        total_ep = max_frame/self.max_steps
        for step in range(len(self.tm_total_scores_each_step)):
            avg_total_score_each_step = [
                total_score/total_ep for total_score in self.tm_total_scores_each_step[step]]
            avg_score_each_step = [
                step_score/total_ep for step_score in self.tm_scores_each_step[step]]
            avg_total.append(avg_total_score_each_step)
            avg_step.append(avg_score_each_step)

        param_str = ''
        for param in sorted(ag_param.keys()):
            param_str += '_' + param
            if str(ag_param[param]) != '':
                param_str += '_' + str(ag_param[param])

        df = pd.DataFrame(data=avg_total)
        plt.plot(df.index, df)
        plt.legend(self.trained_model_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_total_score.jpg"
        plt.savefig(fig_path)

        plt.clf()

        df = pd.DataFrame(data=avg_step)
        plt.plot(df.index, df)
        plt.legend(self.trained_model_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_step_score.jpg"
        plt.savefig(fig_path)
