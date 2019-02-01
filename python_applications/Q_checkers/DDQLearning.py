from DDQNAgent import DDQNAgent
from checkers import game
from gameplay import Gameplay
import random

N_EPISODES = 4000
WIN_REWARD = 10
CAPTURE_REWARD = 0.1
KING_REWARD = 0.3
DRAW_REWARD = 1


class DDQLearning(object):
    def __init__(self):
        self.n_episodes = N_EPISODES
        self.avg_loss = 0
        self.avg_reward = 0
        self.game_play = Gameplay()

    def auto_play(self):
        agents = [DDQNAgent('Smith1'), DDQNAgent('Smith2')]
        turns_hist = {
            agents[0].name: [],
            agents[1].name: []
        }
        for i in range(self.n_episodes):
            random.shuffle(agents)

            moves = [[], []]
            gm = game.Game()
            boardState = self.game_play.update_board_state(gm)
            while (not gm.is_over()):
                curr_agt_idx = gm.whose_turn() - 1
                possible_moves_coord = list(map(lambda x: self.game_play.move_from_pos(x, gm), gm.get_possible_moves()))

                move_idx = self.game_play.get_QAgent_move(agents[curr_agt_idx], boardState, possible_moves_coord)

                move = gm.get_possible_moves()[move_idx]
                moves[curr_agt_idx].append(self.game_play.move_from_pos(move, gm))

                reward = 0
                if (move in gm.board.get_possible_capture_moves()):
                    reward += CAPTURE_REWARD

                new_boardState = self.game_play.make_move(gm, move)

                if (gm.board.searcher.get_piece_by_position(move[1]).king):
                    reward += KING_REWARD

                # Updating previous history
                agt_turns_hist_len = len(turns_hist[agents[curr_agt_idx].name])
                if agt_turns_hist_len > 0:
                    turns_hist[agents[curr_agt_idx].name][agt_turns_hist_len-1]['next_board_state'] = boardState
                    turns_hist[agents[curr_agt_idx].name][agt_turns_hist_len - 1]['next_possible_moves'] = list(map(lambda x: self.game_play.move_from_pos(x, gm), gm.get_possible_moves()))

                # New history
                turns_hist[agents[curr_agt_idx].name].append({
                    'board_state': boardState,
                    'action': self.game_play.move_from_pos(move, gm),
                    'reward': reward,
                    'next_board_state': None,
                    'next_possible_moves': None,
                    'done': False
                })

                boardState = new_boardState

            print("Game Over! ")
            if gm.move_limit_reached():
                print("It's a tie!!")
                for agt in agents:
                    turns_hist[agt.name][len(turns_hist[agt.name]) - 1]['next_board_state'] = boardState
                    turns_hist[agt.name][len(turns_hist[agt.name]) - 1]['next_possible_moves'] = list(map(lambda x: self.game_play.move_from_pos(x, gm), gm.get_possible_moves()))
                    turns_hist[agt.name][len(turns_hist[agt.name]) - 1]['reward'] += DRAW_REWARD
                    turns_hist[agt.name][len(turns_hist[agt.name]) - 1]['done'] = True
            else:
                winner = gm.get_winner() - 1
                print("Winner is: {}".format(agents[winner].name))
                turns_hist[agents[winner].name][len(turns_hist[agents[winner].name]) - 1]['next_board_state'] = boardState
                turns_hist[agents[winner].name][len(turns_hist[agents[winner].name]) - 1]['next_possible_moves'] = list(
                    map(lambda x: self.game_play.move_from_pos(x, gm), gm.get_possible_moves()))
                turns_hist[agents[winner].name][len(turns_hist[agents[winner].name]) - 1]['reward'] += WIN_REWARD
                turns_hist[agents[winner].name][len(turns_hist[agents[winner].name]) - 1]['done'] = True

            for agt in agents:
                agt_turns_hist = turns_hist[agt.name]
                for turn_hist in agt_turns_hist:
                    agt.remember(turn_hist['board_state'],turn_hist['action'],turn_hist['reward'],turn_hist['next_board_state'],turn_hist['next_possible_moves'],turn_hist['done'])

                if(len(agt.memory) > agt.replay_batch_size):
                    agt.replay_memory()


        return agents
