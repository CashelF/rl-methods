from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

class QPolicy(Policy):
    def __init__(self, Q:np.array):
        self.Q = Q
    
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        return float(self.action(state) == action)

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        return np.argmax(self.Q[state])

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    for traj in trajs:
        for t in range(len(traj)):
            tau = t - n + 1
            if tau >= 0:
                G = sum([env_spec.gamma**(i-tau)*traj[i][2] for i in range(tau, min(tau+n, len(traj)))])
                if tau + n < len(traj):
                    G += env_spec.gamma**n * initV[traj[tau+n][0]]
                initV[traj[tau][0]] += alpha * (G - initV[traj[tau][0]])

    return initV

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """
    pi = QPolicy(initQ)
    for traj in trajs:
        T = 2**31 - 1
        t = 0
        tau = 0
        while(tau < T - 1):
            if(t == len(traj)-1):
                T = t + 1
                
            tau = t - n + 1
            if tau >= 0:
                rho = 1
                for i in range(tau+1, min(tau + n, T - 1)+1):
                    rho *= pi.action_prob(traj[i][0], traj[i][1]) / bpi.action_prob(traj[i][0], traj[i][1])
                G = sum([env_spec.gamma**(i-tau)*traj[i][2] for i in range(tau, min(tau+n, T))])
                if tau + n < T:
                    G += env_spec.gamma**n * initQ[traj[tau+n][0], traj[tau+n][1]]
                initQ[traj[tau][0],traj[tau][1]] += alpha * rho * (G - initQ[traj[tau][0], traj[tau][1]])
                pi = QPolicy(initQ)
                
            t += 1

    return initQ, pi