
N_EPISODES = 5000

class DDQLearning(object):
    def __init__(self):
        self.batch_size = 256
        self.n_episodes = N_EPISODES
        self.episodes_count = 0
        self.avg_loss = 0
        self.avg_reward = 0
        self.target_update_threshold = 200


