import gymnasium as gym

from envs.numeric_pddl_wrapper import NumericPDDLWrapper
from rl.rainbow_agent import build_rainbow
from aml.action_model_learner import NumericActionModelLearner
from planner.planner_interface import PlannerInterface
from ramp.ramp_agent import RAMPAgent

env = NumericPDDLWrapper(
    gym.make("NumericPDDLGym-v0")
)

rl_agent = build_rainbow(env)

aml = NumericActionModelLearner()
planner = PlannerInterface()

agent = RAMPAgent(
    env,
    rl_agent,
    aml,
    planner
)

agent.train(episodes=500)
