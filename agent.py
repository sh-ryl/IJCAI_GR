from abc import abstractmethod
from cooperative_craft_world import CooperativeCraftWorldState


def goal_set_to_str(goal_set):
    # *S* why is this function outside of the class?
    result = list(goal_set.keys())[0]
    for i in range(1, len(list(goal_set.keys()))):
        result = result + "_and_" + list(goal_set.keys())[i]

    return result


class Agent(object):

    def __init__(self, name):
        self.name = name


    @abstractmethod
    def perceive(self, reward:float, state:CooperativeCraftWorldState, terminal:bool, is_eval:bool):
        pass


    def reset(self, agent_num, seed, goal_set, externally_visible_goal_sets):
        self.agent_num = agent_num
        self.goal_set = goal_set
        self.externally_visible_goal_sets = externally_visible_goal_sets

        self.goal = list(self.goal_set.keys())[0]
        for i in range(1, len(list(self.goal_set.keys()))):
            self.goal = self.goal + "_and_" + list(self.goal_set.keys())[i]
