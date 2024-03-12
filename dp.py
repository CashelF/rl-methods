from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy

class GreedyPolicy(Policy):
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
    
def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    Q = np.zeros((env.spec.nS,env.spec.nA))
    while(True):
        delta = 0
        for s in range(env.spec.nS):
            v_prev = initV.copy()
            initV[s] = 0
            for a in range(env.spec.nA):
                initV[s] += sum([pi.action_prob(s,a) * (env.TD[s, a, s_] * (env.R[s, a, s_] + env.spec.gamma * v_prev[s_])) for s_ in range(env.spec.nS)])
                Q[s, a] = sum(env.TD[s, a, s_] * (env.R[s, a, s_] + env.spec.gamma * v_prev[s_]) for s_ in range(env.spec.nS))
            delta = max(delta, abs(v_prev[s] - initV[s]))
        if(delta < theta): break

    return initV, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """

    Q = np.zeros((env.spec.nS,env.spec.nA))
    while(True):
        delta = 0
        for s in range(env.spec.nS):
            v = initV[s]
            for a in range(env.spec.nA):
                Q[s, a] = sum([env.TD[s, a, s_] * (env.R[s, a, s_] + env.spec.gamma * initV[s_]) for s_ in range(env.spec.nS)])
            initV[s] = max(Q[s, :])
            delta = max(delta,abs(v - initV[s]))
        if(delta < theta): break

    return initV, GreedyPolicy(Q)
