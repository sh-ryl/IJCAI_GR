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

        # IO Settings
        self.saved_model_dir = saved_model_dir
        self.show_graph = show_graph
        self.first_log_write = True
        self.log_dir = log_dir
        self.probability_plot = Dialog()

        # GR Model Settings
        self.model_temperature = model_temperature
        self.hypothesis_momentum = hypothesis_momentum
        self.kl_tolerance = kl_tolerance
        self.device = torch.device("cuda" if dqn_config.gpu >= 0 else "cpu")

        self.step_number = 0
        self.max_steps = max_steps
        self.goal_list = goal_list

        # Load GR observer models
        self.trained_models = []
        self.tm_paths, self.tm_param = self.find_folders(
            self.saved_model_dir, self.goal_list)

        for i in range(len(self.tm_paths)):
            model_file = f'{self.tm_paths[i]}/model.chk'
            checkpoint = torch.load(model_file, map_location=self.device)

            tm_state_size = checkpoint['model_state_dict']['fc1.weight'].size()[
                1]
            tm_dqn_config = dqn_config
            tm_dqn_config.set_state_size(tm_state_size)

            self.trained_models.append(DQN(tm_dqn_config))
            self.trained_models[i].load_state_dict(
                checkpoint['model_state_dict'])
            self.tm_paths[i] = self.tm_paths[i].removeprefix(
                self.saved_model_dir)

            print(
                f"GR model {i}: {self.tm_paths[i]}, Param: {self.tm_param[i]}")

        # Choose Models to include as observer
        print()
        print("Do you want to include all models listed above?")
        start = input(
            "Press [Enter] to include all\nPress [Other Keys then Enter] to edit ")
        if start != "":
            choice = input(
                "Which models to remove? (Model number separated by comma) ")
            choice = sorted([int(x) for x in choice.split(",")])
            remove_count = 0
            for i in choice:
                print(
                    f"REMOVING GR model: {self.tm_paths[i-remove_count]}")
                self.tm_paths.pop(i-remove_count)
                self.tm_param.pop(i-remove_count)
                self.trained_models.pop(i-remove_count)
                remove_count += 1

            print()
            print("Updated Model List")
            for i in range(len(self.tm_paths)):
                print(
                    f"GR model {i}: {self.tm_paths[i]}, Param: {self.tm_param[i]}")

        # Init scores for evaluation
        self.tm_dkl_sum = [0] * len(self.tm_paths)
        self.tm_dkl_sum_eptotal = [[0 for y in range(
            len(self.tm_paths))]for x in range(max_steps)]  # 100 for max timestep
        self.tm_dkl_step_eptotal = deepcopy(self.tm_dkl_sum_eptotal)

        self.tm_dkl_ravg_prev = [0] * len(self.tm_paths)
        self.tm_dkl_ravg_eptotal = deepcopy(self.tm_dkl_sum_eptotal)
        self.tm_dkl_zbc_eptotal = deepcopy(self.tm_dkl_sum_eptotal)

        self.tm_bi_prob = [
            1.0 / int(len(self.tm_dkl_sum))] * int(len(self.tm_dkl_sum))
        self.tm_bi_prob_eptotal = deepcopy(
            self.tm_dkl_sum_eptotal)

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

    def calculate_action_probs(self, model_no, state, observed_action: int, max_value=100.0):
        state = torch.from_numpy(state.getRepresentation(gr_obs=True,
                                                         gr_param=self.tm_param[model_no])).float().to(
            self.device).unsqueeze(0)
        q = self.trained_models[model_no].forward(
            state).cpu().detach().squeeze()
        probs = F.softmax(q.div(self.model_temperature), dim=0)
        return probs

    def softmax(self, x, temperature):

        x = np.divide(x, temperature)
        # Necessary to ensure that the sum of the values is close enough to 1.
        e_x = np.exp(x - np.max(x)).astype(np.float64)
        result = e_x / e_x.sum(axis=0)
        result = result / result.sum(axis=0, keepdims=1)
        return result

    def perceive(self, state, action: int, frame_num, print_result, momentum=0.95):
        bayes_pr_act = 0
        tm_act_probs = []
        tm_dkl_step = []
        for i in range(len(self.trained_models)):
            act_probs = self.calculate_action_probs(i, state, action)
            tm_act_probs.append(act_probs)

            # kl div sum
            dkl_max = 100.0
            dkl_step = min(act_probs[action].pow(-1).log().item(), dkl_max)
            tm_dkl_step.append(dkl_step)

            # bayes prob
            bayes_pr_act += self.tm_bi_prob[i] * act_probs[action]

        tm_bi_prob_new = [0] * len(self.trained_models)
        eps = 1e-20
        if print_result:
            print()
            print("---------------- GR Inference ---------------")
            print(
                f"{'No.':3} {'Trained Model Paths':60} {'Sum DKL':10} {'Run DKL':10} {'ZBC DKL':10} {'BI Prob':10} Action Probabilities")
            print()

        for i in range(len(self.trained_models)):
            # kl div running avg
            dkl_ravg = momentum * \
                self.tm_dkl_ravg_prev[i] + (1 - momentum) * tm_dkl_step[i]

            # kl div zero bias corrected
            denom = 1 - pow(momentum, ((frame_num % self.max_steps) + 1))
            dkl_zbc = dkl_ravg/denom

            # bayesian inf
            tm_bi_prob_new[i] = tm_act_probs[i][action] * \
                self.tm_bi_prob[i] / bayes_pr_act+eps

            if print_result:
                tm_act_probs[i] = [round(float(x), 3) for x in tm_act_probs[i]]
                a = str(i)+'.'
                b = self.tm_paths[i]
                c = round(self.tm_dkl_sum[i], 3)
                d = round(dkl_ravg, 3)
                e = round(dkl_zbc, 3)
                f = round(float(tm_bi_prob_new[i]), 3)
                g = tm_act_probs[i]
                print(
                    f"{a:3} {b:60} {c:<10} {d:<10} {e:<10} {f:<10} {g}")

            # update scores
            # dkl sum
            self.tm_dkl_sum[i] += tm_dkl_step[i]
            self.tm_dkl_sum_eptotal[frame_num %
                                    self.max_steps][i] += self.tm_dkl_sum[i]
            self.tm_dkl_step_eptotal[frame_num %
                                     self.max_steps][i] += tm_dkl_step[i]

            # dkl ravg
            self.tm_dkl_ravg_prev[i] = dkl_ravg
            self.tm_dkl_ravg_eptotal[frame_num %
                                     self.max_steps][i] += dkl_ravg

            # dkl zbc
            self.tm_dkl_zbc_eptotal[frame_num %
                                    self.max_steps][i] += dkl_zbc
            # bi prob
            self.tm_bi_prob_eptotal[frame_num %
                                    self.max_steps][i] += tm_bi_prob_new[i]
        self.tm_bi_prob = tm_bi_prob_new

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
                    weight_list = [float(x[1]) for x in matches]

                if "level" in obj_list:
                    obj_list.remove("level")

                if "discount" in obj_list:
                    i = obj_list.index("discount")
                    obj_list.pop(i)
                    weight_list.pop(i)

                if set(obj_list) == set(required_objects):
                    tm_param = {}
                    tm_goal_dic = {}

                    for param in param_set.intersection(base_folder):
                        tm_param[param] = ''

                    if "uvfa" in tm_param:
                        tm_param["uvfa"] = obj_list

                        # get goal dic
                        for obj in required_objects:
                            tm_goal_dic[obj] = 0
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

        # Extract sorted combined tuples
        trained_models_goal_dic, trained_models_path, trained_models_param = zip(
            *sorted_combined)

        return list(trained_models_path), list(trained_models_param)

    def reset(self):
        self.tm_dkl_sum = [0] * len(self.tm_paths)
        self.tm_bi_prob = [
            1.0 / int(len(self.tm_dkl_sum))] * int(len(self.tm_dkl_sum))
        self.tm_dkl_ravg_prev = [0] * len(self.tm_paths)

    def get_inference(self):
        temp_min = max(self.tm_dkl_sum)
        temp_min_id = 0

        for i in range(len(self.tm_dkl_sum)):
            if self.tm_dkl_sum[i] < temp_min:
                temp_min = self.tm_dkl_sum[i]
                temp_min_id = i

        return temp_min_id

    def get_result(self, max_frame, goal_str, ag_param=dict()):
        avg_dkl_sum = []
        avg_dkl_step = []
        avg_dkl_ravg = []
        avg_dkl_zbc = []
        avg_bi_prob = []

        ep_num = max_frame/self.max_steps
        for step in range(len(self.tm_dkl_sum_eptotal)):
            # savg = step avg
            savg_dkl_sum = [
                ep_total/ep_num for ep_total in self.tm_dkl_sum_eptotal[step]]
            savg_dkl_step = [
                ep_total/ep_num for ep_total in self.tm_dkl_step_eptotal[step]]
            savg_dkl_ravg = [
                ep_total/ep_num for ep_total in self.tm_dkl_ravg_eptotal[step]]
            savg_dkl_zbc = [
                ep_total/ep_num for ep_total in self.tm_dkl_zbc_eptotal[step]]
            savg_bi_prob = [
                ep_total/ep_num for ep_total in self.tm_bi_prob_eptotal[step]]

            avg_dkl_sum.append(savg_dkl_sum)
            avg_dkl_step.append(savg_dkl_step)
            avg_dkl_ravg.append(savg_dkl_ravg)
            avg_dkl_zbc.append(savg_dkl_zbc)
            avg_bi_prob.append(savg_bi_prob)

        param_str = ''
        for param in sorted(ag_param.keys()):
            param_str += '_' + param
            if str(ag_param[param]) != '':
                param_str += '_' + str(ag_param[param])

        # dkl sum
        df = pd.DataFrame(data=avg_dkl_sum)
        plt.plot(df.index, df)
        plt.legend(self.tm_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_avg_dkl_sum.jpg"
        plt.savefig(fig_path)

        plt.clf()

        # dkl step
        df = pd.DataFrame(data=avg_dkl_step)
        plt.plot(df.index, df)
        plt.legend(self.tm_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_avg_dkl_step.jpg"
        plt.savefig(fig_path)

        plt.clf()

        # dkl ravg
        df = pd.DataFrame(data=avg_dkl_ravg)
        plt.plot(df.index, df)
        plt.legend(self.tm_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_avg_dkl_ravg.jpg"
        plt.savefig(fig_path)

        plt.clf()

        # dkl zbc
        df = pd.DataFrame(data=avg_dkl_zbc)
        plt.plot(df.index, df)
        plt.legend(self.tm_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_avg_dkl_zbc.jpg"
        plt.savefig(fig_path)

        plt.clf()

        # bi prob
        df = pd.DataFrame(data=avg_bi_prob)
        plt.plot(df.index, df)
        plt.legend(self.tm_paths)

        fig_path = self.log_dir + "/" + goal_str + param_str + "_avg_bi_prob.jpg"
        plt.savefig(fig_path)
