from utils.learning_utils import LearningUtils
from utils.gameplay import Gameplay
from tqdm import tqdm
import collections, numpy

n_games = 50

agent_v1 = LearningUtils.load_agent('agt_{}.h5'.format(LearningUtils.V1_NAME), version=1, with_eps=False)
agent_v2 = LearningUtils.load_agent('agt_{}.h5'.format(LearningUtils.V2_NAME), version=2, with_eps=False)

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

