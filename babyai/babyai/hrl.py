"""
Note: This file is deprecated, but is retained for development reference.
"""

import math
import operator
from functools import reduce
from torch.multiprocessing import Process, Pipe
import torch

import numpy as np
import gym
from tqdm import tqdm
from gym import error, spaces, utils
from babyai.utils.agent import ModelAgent
import logging
logger = logging.getLogger(__name__)
import concurrent.futures
from copy import deepcopy

logger.setLevel(logging.WARNING)


def multi_worker(conn, envs):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            ret = []
            for env, a, stopped in zip(envs, data[0], data[1]):
                if not stopped:
                    obs, reward, done, info = env.step(a)
                    if done:
                        obs = env.reset()
                    ret.append((obs, reward, done, info))
                else:
                    ret.append((None, 0, False, None))
            conn.send(ret)
        elif cmd == "reset":
            ret = []
            for env in envs:
                ret.append(env.reset())
            conn.send(ret)
        elif cmd == "render_one":
            mode, highlight = data
            ret = envs[0].render(mode, highlight)
            conn.send(ret)
        elif cmd == "__str__":
            ret = str(envs[0])
            conn.send(ret)
        else:
            raise NotImplementedError


class HRLManyEnvs(gym.Env):

    def __init__(self, envs, hrl="", pi_l=None, done_classifier=False, instr_handler=None, N=None, T=None, use_procs=True, envs_per_proc=64, oracle_rate=1,
            reward_shaping=None, subtask_model=None, subtask_model_preproc=None, subtask_dataset=None):
        assert len(envs) >= 1, "No environment given."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spec = envs[0].spec
        self.use_procs = use_procs
        self.envs_per_proc = envs_per_proc
        self.envs = envs
        self.num_envs = len(self.envs)
        self.env_name = self.envs[0].unwrapped.spec.id
        self.hrl = hrl
        self.pi_l = pi_l
        self.N = N
        self.T = T
        self.ts = np.array([0 for _ in range(self.num_envs)])
        self.hs = np.array([0 for _ in range(self.num_envs)])
        self.instr_handler = instr_handler
        self.oracle_dist = np.full(self.num_envs, oracle_rate)
        self.reward_shaping = reward_shaping
        self.subtask_model = subtask_model
        self.subtask_model_preproc = subtask_model_preproc
        self.subtask_dataset = subtask_dataset
        self.done_classifier = done_classifier
        if self.done_classifier:
            self.done_action = 1
        else:
            self.done_action = self.envs[0].actions.done

        A = self.envs[0].action_space.n
        self.has_option = False
        self.has_offset = False
        self.pi_l_option = None
        self.pi_l_offset = None
        if self.hrl in ["strict_hrl_random", "strict_hrl_oracle", "strict_hrl_projection"]:
            # Action space: ["follow pi_l"]
            self.pi_l_option = 0
            self.has_option = True
        elif self.hrl in ["soft_hrl_random", "soft_hrl_oracle", "soft_hrl_projection"]:
            # Action space: primitives + ["follow pi_l"]
            self.pi_l_option = A
            self.has_option = True
        elif self.hrl in ["strict_hrl"]:
            # Action space: instructions
            self.pi_l_offset = 0
            self.has_offset = True
        elif self.hrl in ["soft_hrl"]:
            # Action space: primitives + instructions
            self.pi_l_offset = A
            self.has_offset = True

        if self.has_option or self.reward_shaping is not None:
            self.stacks = [[] for _ in range(self.num_envs)]
        
        self.succeeded = np.array([False for _ in range(self.num_envs)])
        
        if self.reward_shaping in ["pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]:
            self.pi_l_already_done = np.array([[False for j in range(self.instr_handler.D_l_size())] for i in range(self.num_envs)])
            self.pi_l_all_already_done = np.array([[False for j in range(self.instr_handler.D_l_size())] for i in range(self.num_envs)])

        self.obss = []
        self.locals = []
        self.processes = []
        self.start_procs()
    
    def __len__(self):
        return self.num_envs

    def gen_obs(self):
        return self.obss
    
    def render(self, mode="rgb_array", highlight=False):
        self.locals[0].send(("render_one", (mode, highlight)))
        return self.locals[0].recv()

    def __str__(self):
        self.locals[0].send(("__str__", None))
        return self.locals[0].recv()

    def start_procs(self):
        if self.use_procs:
            logger.info("spinning up {} processes".format(self.num_envs))
            for i in range(0, self.num_envs, self.envs_per_proc):
                local, remote = Pipe()
                self.locals.append(local)
                p = Process(target=multi_worker, args=(remote, self.envs[i:i+self.envs_per_proc]))
                p.daemon = True
                p.start()
                remote.close()
                self.processes.append(p)
            logger.info("done spinning up processes")

    def request_reset(self):
        logger.info("requesting resets")
        if self.use_procs:
            for local in self.locals:
                local.send(("reset", None))
            self.obss = []
            logger.info("requested resets")
            for local in self.locals:
                self.obss += local.recv()
        else:
            self.obss = []
            for i in range(self.num_envs):
                self.obss.append(self.envs[i].reset())
        logger.info("completed resets")

    def reset(self):
        self.request_reset()
        self.ts *= 0
        self.hs *= 0
        return [obs for obs in self.obss]

    def request_step(self, primitive_actions, stop_mask):
        if self.use_procs:
            for i in range(0, self.num_envs, self.envs_per_proc):
                self.locals[i//self.envs_per_proc].send(("step", [primitive_actions[i:i+self.envs_per_proc], stop_mask[i:i+self.envs_per_proc]]))
            results = []
            for i in range(0, self.num_envs, self.envs_per_proc):
                res = self.locals[i//self.envs_per_proc].recv()
                for j in range(len(res)):
                    results.append(res[j])
                    if results[-1][0] != None:
                        self.obss[i + j] = results[-1][0]
        else:
            results = []
            for i, (a, stopped) in enumerate(zip(primitive_actions, stop_mask)):
                if not stopped:
                    obs, reward, done, info = self.envs[i].step(a.item())
                    if done:
                        obs = self.envs[i].reset()
                    results.append((obs, reward, done, info))
                    self.obss[i] = results[i][0]
                else:
                    results.append((None, 0, False, None))
        return zip(*results)

    def pop_all(self, stacks):
        stacks = [stacks[i][1:] if len(stacks[i]) > 1 else stacks[i] for i in range(len(stacks))]
        return stacks

    def pop_masked(self, stacks, mask, allow_zero=False):
        if allow_zero:
            stacks = [stacks[i][1:] if len(stacks[i]) > 0 and mask[i] else stacks[i] for i in range(len(stacks))]
        else:
            stacks = [stacks[i][1:] if len(stacks[i]) > 1 and mask[i] else stacks[i] for i in range(len(stacks))]
        return stacks

    def pop(self, stack, allow_zero=False):
        if allow_zero:
            stack = stack[1:] if len(stack) > 0 else stack
        else:
            stack = stack[1:] if len(stack) > 1 else stack
        return stack

    def reset_pi_l(self):
        self.pi_l.analyze_feedback(None, 1)
        self.pi_l.on_reset()

    def reset_pi_l_partial(self, reset_mask):
        self.pi_l.analyze_feedback(None, torch.tensor(reset_mask).to(self.device).int().unsqueeze(1))

    def step(self, actions, add_h=False, reward_shaping=False, extra_info=False, project=False):
        if type(actions) != np.ndarray:
            if type(actions) == list or type(actions) == int:
                actions = np.array(actions)
            elif type(actions) == torch.Tensor:
                actions = actions.cpu().numpy()
            else:
                raise TypeError
        actions_to_take = actions.copy()

        if self.oracle_dist[0] < 1:
            nullify_oracle = np.random.binomial(1 , 1-self.oracle_dist)
            actions_to_take -= (nullify_oracle * (actions_to_take == self.pi_l_option))
            add_h_arr = (self.hs == 0)
        if add_h and np.any(add_h_arr):
            self.step(add_h_arr * 7 + np.invert(add_h_arr) * -1, add_h=False)

        if self.has_option:
            high_level = actions_to_take == self.pi_l_option
        elif self.has_offset:
            high_level = actions_to_take >= self.pi_l_offset
        else:
            high_level = actions_to_take < 0
        primitive = high_level == False

        done_mask = np.array([False for _ in range(self.num_envs)])
        reward_agg = np.zeros(self.num_envs)
        if high_level.any(): # For now, we'll assume an option
            stop_mask = primitive
            if self.hrl.endswith('projection') and ((self.ts == 0) & high_level).any():
                old_missions, proj_ind = zip(*[(self.obss[i]['mission'], i) for i in range(len(self.obss)) if high_level[i] and self.ts[i] == 0])
                proj_stacks = self.instr_handler.get_projection_stacks(old_missions)
                for i, j in enumerate(proj_ind):
                    self.stacks[j] = proj_stacks[i]
            if self.has_option:
                for i in range(self.num_envs):
                # If ts is 0 for an env, we'll assume the mission is up to date but the 
                # stack is not. Calculate its new stack.
                    if high_level[i] and (self.ts[i] == 0 or len(self.stacks[i]) == 0 or self.hrl.endswith('random')):
                        if self.hrl.endswith('oracle'):
                            old_mission = self.obss[i]['mission']
                            self.stacks[i] = self.instr_handler.get_oracle_stack(old_mission, self.env_name)
                        elif self.hrl.endswith('random'):
                            self.stacks[i] = [self.instr_handler.get_random_instruction()]
            # Run everything out for T mini-steps, but we might stop updating some envs.
            for _ in range(self.T):
                # Update all observations to contain their latest stack mission.
                for i in range(len(self.obss)):
                    if not stop_mask[i]:
                        if self.has_option:
                            self.obss[i]['mission'] = self.stacks[i][0]
                        if self.has_offset:
                            self.obss[i]['mission'] = self.instr_handler.get_instruction(actions_to_take[i] - self.pi_l_offset)
                        if project:
                            self.obss[i]['image'][3][6] = np.array([1, 0, 0])
                # Run pi_l on all the observations.
                primitive_actions = self.pi_l.act_batch(self.obss, stop_mask=stop_mask)['action'].cpu().numpy()
                # Execute those steps on the envs (unless stop_mask is true).
                obs, reward, done, info = self.request_step(primitive_actions, stop_mask)
                # Aggregate any reward we saw.
                reward = np.array(reward)
                reward_agg += reward
                # Aggregate any done envs.
                done = np.array(done)
                done_mask |= done
                # Stop executing on anything done, or where pi_l predicted done.
                stop_mask |= (primitive_actions == self.done_action)
                stop_mask |= done
                # Stop early if we don't need to execute anything else.
                if stop_mask.all():
                    # logger.info("stopping early")
                    break
                self.ts[stop_mask == False] += 1
                self.hs[stop_mask == False] += 1
            # pi_l should forget its history.
            self.reset_pi_l()

        # Soft->primitive or Non-HRL
        if primitive.any():
            stop_mask = (actions_to_take == self.pi_l_option) | (actions_to_take < 0)

            if self.reward_shaping == "pi_l" or self.reward_shaping == "pi_l_complete" \
                or self.reward_shaping == "pi_l_complete_one" or self.reward_shaping == "pi_l_complete_one_discounted":
                # Assuming stop_mask = False everywhere
                self.pi_l_obss = deepcopy(self.obss)
                self.out_of_instr = np.array([False for _ in range(self.num_envs)])
                for i in range(self.num_envs):
                    if self.hs[i] == 0: # Just reset
                        old_mission = self.pi_l_obss[i]['mission']
                        self.stacks[i] = self.instr_handler.get_oracle_stack(old_mission, self.env_name)
                    if len(self.stacks[i]) > 0:
                        self.pi_l_obss[i]['mission'] = self.stacks[i][0]
                        if project:
                            self.pi_l_obss[i]['image'][3][6] = np.array([1, 0, 0]) 
                    else:
                        self.out_of_instr[i] = True
                # add oracle instructions
                pi_l_eval = self.pi_l.act_batch(self.pi_l_obss, stop_mask=self.out_of_instr)
                pi_l_probs = pi_l_eval['dist'].probs[:,actions_to_take].diag().cpu().numpy() * (1-self.out_of_instr)
                pi_l_actions = pi_l_eval['action'].cpu().numpy() * (1-self.out_of_instr) - self.out_of_instr.astype(int)
                pi_l_done = pi_l_actions == self.done_action
            elif self.reward_shaping == "pi_l_all" or self.reward_shaping == "pi_l_complete_subtask_pred" \
                or self.reward_shaping == "pi_l_complete_subtask_pred_once" or self.reward_shaping == "pi_l_complete_subtask_online_once"\
                    or self.reward_shaping == "pi_l_complete_subtask_online_once_max":
                self.pi_l_obss = [deepcopy(self.obss[i]) for i in range(self.num_envs) for _ in range(self.instr_handler.D_l_size())]
                for i in range(self.num_envs):
                    if self.hs[i] == 0: # Just reset
                        old_mission = self.obss[i]['mission']
                        self.stacks[i] = self.instr_handler.get_oracle_stack(old_mission)
                        if self.reward_shaping in ["pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]:
                            if "online" in self.reward_shaping:
                                if self.succeeded[i] == True and self.pi_l_all_already_done[i].sum() > 0:
                                    self.subtask_dataset.add_demos([(old_mission, [(-1,np.where(self.pi_l_all_already_done[i])[0])])])
                            self.pi_l_already_done[i] *= False
                            self.pi_l_all_already_done[i] *= False
                            self.succeeded[i] = False
                    for j in range(self.instr_handler.D_l_size()):
                        self.pi_l_obss[i*self.instr_handler.D_l_size() + j]['mission'] = self.instr_handler.get_instruction(j)
                        if project:
                            self.pi_l_obss[i*self.instr_handler.D_l_size() + j]['image'][3][6] = np.array([1,0,0])
                pi_l_eval = self.pi_l.act_batch(self.pi_l_obss, stop_mask=None)
                pi_l_actions = pi_l_eval['action'].cpu().numpy()
                pi_l_done = pi_l_actions == self.done_action
                pi_l_done = pi_l_done.reshape((self.num_envs, self.instr_handler.D_l_size()))
                if self.reward_shaping in ["pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]:
                    pi_l_done *= np.invert(self.pi_l_already_done)
                if self.reward_shaping in ["pi_l_complete_subtask_pred", "pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]\
                    and pi_l_done.sum() > 0:
                    done_instr_idx, done_sub_idx = np.where(pi_l_done==True)
                    done_instr_text = [self.obss[i]['mission'] for i in done_instr_idx]
                    done_sub_text = self.instr_handler.missions[done_sub_idx]
                    done_instr_proc = self.subtask_model_preproc(done_instr_text)
                    done_sub_proc = self.subtask_model_preproc(done_sub_text)
                    if self.reward_shaping in ["pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]:
                        done_instr_proc = done_instr_proc.to(self.device)
                        done_sub_proc = done_sub_proc.to(self.device)
                    predicted_subtasks = self.subtask_model(done_instr_proc, done_sub_proc).round().detach().cpu().numpy().astype(bool)
                    self.pi_l_all_already_done ^= pi_l_done
                    pi_l_done &= False
                    if self.reward_shaping in ["pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]:
                        for j in range(len(done_instr_idx)):
                            if predicted_subtasks[j]:
                                pi_l_done[done_instr_idx[j], done_sub_idx[j]] = True
                                self.pi_l_already_done[done_instr_idx[j], done_sub_idx[j]] = True
                    else:
                        for j in range(len(done_instr_idx)):
                            pi_l_done[done_instr_idx[j], done_sub_idx[j]] = predicted_subtasks[j]

            obs, reward, done, info = self.request_step(actions_to_take, stop_mask)
            reward = np.array(reward)
            reward_agg += reward
            done = np.array(done)
            done_mask |= done

            if self.reward_shaping == "pi_l":
                self.stacks = self.pop_masked(self.stacks, pi_l_done, allow_zero=True)
                self.reset_pi_l_partial(pi_l_done | done)
                info = (pi_l_probs, torch.tensor(pi_l_actions).to(self.device))
            elif self.reward_shaping == "pi_l_complete":
                self.stacks = self.pop_masked(self.stacks, pi_l_done, allow_zero=True)
                self.reset_pi_l_partial(pi_l_done | done)
                info = (pi_l_done.astype(int), torch.tensor(pi_l_actions).to(self.device))
            elif self.reward_shaping == "pi_l_complete_one":
                self.stacks = self.pop_masked(self.stacks, pi_l_done, allow_zero=True)
                self.reset_pi_l_partial(pi_l_done | done)
                info = (np.array([1 if len(self.stacks[i]) == 1 and pi_l_done[i]\
                    else 0 for i in range(self.num_envs)]), torch.tensor(pi_l_actions).to(self.device))
            elif self.reward_shaping == "pi_l_complete_one_discounted":
                self.stacks = self.pop_masked(self.stacks, pi_l_done, allow_zero=True)
                self.reset_pi_l_partial(pi_l_done | done)
                info = (np.array([1-self.hs[i]/128 if len(self.stacks[i]) == 1 and pi_l_done[i]\
                    else 0 for i in range(self.num_envs)]), torch.tensor(pi_l_actions).to(self.device))
            elif self.reward_shaping == "pi_l_all":
                bonus = np.zeros(self.num_envs)
                penalty = np.zeros(self.num_envs)
                for i, env_pi_l_done_by_instr in enumerate(pi_l_done):
                    should_pop = False
                    for j, l_done in enumerate(env_pi_l_done_by_instr):
                        if l_done:
                            if len(self.stacks[i]) > 0 and self.instr_handler.get_instruction(j) == self.stacks[i][0]:
                                should_pop = True
                                bonus[i] += 1
                            else:
                                penalty[i] += 1
                    if should_pop:
                        self.stacks[i] = self.pop(self.stacks[i], allow_zero=True)
                self.reset_pi_l_partial((pi_l_done.sum(1) > 0).repeat(self.instr_handler.D_l_size()))
                info = (np.stack((bonus, penalty), axis=1), torch.tensor(pi_l_actions).to(self.device))
                env_extra_info = pi_l_eval
            if self.reward_shaping in ["pi_l_complete_subtask_pred", "pi_l_complete_subtask_pred_once", "pi_l_complete_subtask_online_once", "pi_l_complete_subtask_online_once_max"]:
                self.reset_pi_l_partial((pi_l_done.sum(1) > 0).repeat(self.instr_handler.D_l_size()))
                info = (pi_l_done.sum(1).clip(0, 1), torch.tensor(pi_l_actions).to(self.device))
            self.hs[stop_mask == False] += 1
        # Anything done should be a reset env, so we'll set its ts to 0.
        self.hs[done_mask] *= 0
        self.ts[done_mask] *= 0
        self.succeeded = reward_agg > 0
        if self.hrl.endswith('oracle') or self.hrl.endswith('projection'):
            # Pop off the stacks if there's more than one instruction.
            self.stacks = self.pop_masked(self.stacks, high_level)
        
        if extra_info:
            return [obs for obs in self.obss], reward_agg, done_mask, info, env_extra_info
        return [obs for obs in self.obss], reward_agg, done_mask, info

    def __del__(self):
        for p in self.processes:
            p.terminate()

class Visit2Wrapper(gym.core.ObservationWrapper):
    def __init__(self, env, tile_size=8):
        super().__init__(env)
    def observation(self, obs):
        old_mission_words = obs['mission'].split(' ')
        obs['mission'] = ' '.join(['visit', *old_mission_words[2:6], *old_mission_words[8:]])
        return obs