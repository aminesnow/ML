from v1.DDQAgent import DDQAgent as DDQAgentv1
from v2.DDQAgent import DDQAgent as DDQAgentv2


class AgentUtils(object):

    V1_NAME = 'Smith'
    V2_NAME = 'Samantha'

    @staticmethod
    def train_agent(n_episodes, start_eps=1.0, agt=None, version=1, resume=False):
        if resume:
            agt.epsilon = start_eps
        else:
            if version == 1:
                agt = DDQAgentv1(AgentUtils.V1_NAME, True)
            else:
                agt = DDQAgentv2(AgentUtils.V2_NAME, True)
        return agt.auto_play(n_episodes)

    @staticmethod
    def load_agent(weights_name, version=1, with_eps=False):
        if version == 1:
            agt = DDQAgentv1(AgentUtils.V1_NAME, with_eps)
        else:
            agt = DDQAgentv2(AgentUtils.V2_NAME, with_eps)
        agt.load_weights(weights_name)
        return agt
