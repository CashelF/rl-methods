from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    Q = initQ
    C = np.zeros((env_spec.nS,env_spec.nA))
    for traj in trajs:
        G = 0
        W = 1
        for t in range(len(traj)-1,-1,-1):
            s, a, r, s_ = traj[t]
            G = env_spec.gamma * G + r
            C[s,a] += 1
            Q[s,a] += (W/C[s][a]) * (G - Q[s,a])
            if bpi.action_prob(s,a) == 0:
                break
            W *= pi.action_prob(s,a) / bpi.action_prob(s,a)

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    C = np.zeros((env_spec.nS,env_spec.nA))
    Q = initQ
    for traj in trajs:
        G = 0
        W = 1
        for t in range(len(traj)-1,-1,-1):
            s, a, r, s_ = traj[t]
            G = env_spec.gamma * G + r
            C[s,a] += W
            Q[s,a] += (W/C[s,a]) * (G - Q[s,a])
            if bpi.action_prob(s,a) == 0:
                break
            W *= pi.action_prob(s,a) / bpi.action_prob(s,a)
            if W == 0:
                break

    return Q
