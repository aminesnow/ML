from utils.agent_utils import AgentUtils
from utils.gameplay import Gameplay
from tqdm import tqdm
import numpy

n_games = 100
eps = 0.01
with_eps = True

agent_v1 = AgentUtils.load_agent('agt_{}_test.h5'.format(AgentUtils.V1_NAME), version=1, with_eps=with_eps)
agent_v2 = AgentUtils.load_agent('agt_{}_test.h5'.format(AgentUtils.V2_NAME), version=2, with_eps=with_eps)
agent_v1.epsilon = eps
agent_v2.epsilon = eps

results = []
for i in tqdm(range(n_games)):
    res = Gameplay.run_agent_duel(agent_v1, agent_v2, verbose=False)
    results.append(res)

results = numpy.array(results)

agt1_wins = (results == agent_v1.name).sum()
agt2_wins = (results == agent_v2.name).sum()
ties = (results == 'tie').sum()

print('{} wins: {}'.format(agent_v1.name, agt1_wins))
print('{} wins: {}'.format(agent_v2.name, agt2_wins))
print('Ties: {}'.format(ties))

