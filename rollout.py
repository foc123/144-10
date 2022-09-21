#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import IPython
import numpy as np
import matplotlib.pyplot as plt
import gym
import ray
#from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env

#from bag_deep_ckt.autockt.envs.bag_opamp_discrete import TwoStageAmp
#from envs.ngspice_vanilla_opamp import TwoStageAmp
from decap_env import *

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
# Note: if you use any custom models or envs, register them here first, e.g.:
#
# ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
# register_env("pa_cartpole", lambda _: ParametricActionCartpole(10))
register_env("decap-v0", lambda config:DecapPlaceParallel(config))

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    parser.add_argument(
        "--num_val_specs",
        type=int,
        default=50,
        help="Number of untrained objectives to test on")
    parser.add_argument(
        "--traj_len",
        type=int,
        default=30,
        help="Length of each trajectory")
    return parser


def savefig(capidx, rollout_step):
    res_array = readresult(str(os.getpid())+'_chiplet1_vdd_1_vdi.csv')
    x = res_array[:, 0]
    y = res_array[:, 1]
    z = res_array[:, 2]
    vdi = z.sum()
    fig, (ax2) = plt.subplots(nrows=1)
    ax2.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr2 = ax2.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax2)
    ax2.plot(x, y, 'ko', ms=3)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    plt.subplots_adjust(hspace=0.5)
    plt.title('VDI={},cap={}'.format(vdi, capidx))
    plt.savefig('./result_fig/'+str(os.getpid())+'_VDI_distri_'+ str(rollout_step))
    plt.close()


def run(args, parser):
    config = args.config
    if not config:
        # Load configuration from file
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.json")
        if not os.path.exists(config_path):
            raise ValueError(
                "Could not find params.json in either the checkpoint dir or "
                "its parent directory.")
        with open(config_path) as f:
            config = json.load(f)
        if "num_workers" in config:
            config["num_workers"] = min(2, config["num_workers"])

    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init(num_cpus=7)

    #cls = get_agent_class(args.run)
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    rollout(agent, args.env, num_steps, args.out, args.no_render)
    ray.shutdown()

def rollout(agent, env_name, num_steps, out="assdf", no_render=True):
    #if hasattr(agent, "local_evaluator"):
        #env = agent.local_evaluator.env
    env_config = {} #{"generalize":True,"num_valid":args.num_val_specs, "save_specs":False, "run_valid":True}
    if env_name == "decap-v0":
        env = DecapPlaceParallel(env_config=env_config)
    else:
        env = gym.make(env_name)

    #get unnormlaized specs
    #norm_spec_ref = env.global_g
    #spec_num = len(env.specs)
    ideal_spec =0.9e-10
     
    if hasattr(agent, "local_evaluator"):
        state_init = agent.local_evaluator.policy_map[
            "default"].get_initial_state()
    else:
        state_init = []
    if state_init:
        use_lstm = True
    else:
        use_lstm = False
    #state_init = []
    rollouts = []
    next_states = []
    obs_reached = []
    obs_nreached = []
    action_array = []
    action_arr_comp = []
    rollout_steps = 0
    reached_spec = 0
    while rollout_steps < args.num_val_specs:
        if out is not None:
            rollout_num = []
        state = env.reset()
        
        done = False
        reward_total = 0.0
        steps=0
        while not done and steps < args.traj_len:
            if use_lstm:
                action, state_init, logits = agent.compute_action(
                    state, state=state_init)
            else:
                action = agent.compute_action(state)
                action_array.append(action)

            next_state, reward, done, _ = env.step(action)
            print(action)
            print(reward)
            print(done)
            reward_total += reward
            if not no_render:
                env.render()
            if out is not None:
                rollout_num.append(reward)
                next_states.append(next_state)
            steps += 1
            state = next_state
        if done == True:
            reached_spec += 1
            obs_reached.append(ideal_spec)
            action_arr_comp.append(action_array)
            action_array = []
            #pickle.dump(action_arr_comp, open("action_arr_test", "wb"))
        else:
            obs_nreached.append(ideal_spec)          #save unreached observation 
            action_array=[]
        if out is not None:
            rollouts.append(rollout_num)
        #print("Episode reward", reward_total)
        savefig(state[-10:],rollout_steps) 
        rollout_steps+=1
        #if out is not None:
            #pickle.dump(rollouts, open(str(out)+'reward', "wb"))
        #pickle.dump(obs_reached, open("opamp_obs_reached_test","wb"))
        #pickle.dump(obs_nreached, open("opamp_obs_nreached_test","wb"))
        #print("Specs reached: " + str(reached_spec) + "/" + str(len(obs_nreached)))

    #print("Num specs reached: " + str(reached_spec) + "/" + str(args.num_val_specs))

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
