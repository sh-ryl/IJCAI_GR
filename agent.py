from abc import abstractmethod
from cooperative_craft_world import CooperativeCraftWorldState


class Agent(object):

    def __init__(self, name):
        self.name = name


    @abstractmethod
    def perceive(self, reward:float, state:CooperativeCraftWorldState, terminal:bool, is_eval:bool, model_file:str):
        pass


    def reset(self, goal_set, externally_visible_goal_sets):
        # this function is used in goal_recogniser.py!
        # to be investigated: but also why is this here?
        # this current setup assumes that goal is agent specific
        self.goal_set = goal_set
        self.externally_visible_goal_sets = externally_visible_goal_sets

        self.goal = list(self.goal_set.keys())[0]
        for i in range(1, len(list(self.goal_set.keys()))):
            self.goal = self.goal + "_and_" + list(self.goal_set.keys())[i]
