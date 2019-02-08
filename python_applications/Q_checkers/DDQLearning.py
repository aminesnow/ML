from DDQNAgent import DDQNAgent
from checkers import game
from gameplay import Gameplay
import numpy as np

WIN_REWARD = 1
CAPTURE_REWARD = 0.05
KING_REWARD = 0.075
DRAW_REWARD = 0.25


class DDQLearning(object):
    def __init__(self):
        self.game_play = Gameplay()
        self.agent = DDQNAgent('Smith', True)

    def auto_play(self, n_episodes):
        for i in range(n_episodes):
            print("Episode {}".format(i))
            turns_hist = {
                1: [],
                2: []
            }
            gm = game.Game()
            boardState = self.game_play.board_state_from_board(gm.board)

            while (not gm.is_over()):
                player = gm.whose_turn()

                possible_board_states = self.game_play.board_states_from_possible_moves(gm.board)
                #move_idx, q_val = self.game_play.get_QAgent_move(self.agent, boardState, gm.board, (player == 2))
                move_idx, q_val = self.game_play.get_QAgent_move_pp(self.agent, gm.board)

                if (player == 2):
                    boardState = Gameplay.invert_board(boardState)
                    possible_board_states = np.array(list(map(lambda x: Gameplay.invert_board(x), possible_board_states)))

                # Updating previous history
                if len(turns_hist[player]) > 0:
                    turns_hist[player][-1]['next_board_state'] = boardState
                    turns_hist[player][-1]['next_possible_board_states'] = possible_board_states

                move = gm.get_possible_moves()[move_idx]

                reward = 0
                if (move in gm.board.get_possible_capture_moves()):
                    reward += CAPTURE_REWARD

                piece_was_king = gm.board.searcher.get_piece_by_position(move[0]).king
                new_boardState = self.game_play.make_move(gm, move)

                if (not piece_was_king) and gm.board.searcher.get_piece_by_position(move[1]).king:
                    reward += KING_REWARD

                if len(turns_hist[self.get_other_player(player)]) > 0:
                    turns_hist[self.get_other_player(player)][-1]['reward'] -= reward

                # New history
                turns_hist[player].append({
                    'board_state': boardState,
                    'board_state_action': new_boardState,
                    'reward': reward,
                    'next_board_state': None,
                    'next_possible_board_states': None,
                    'done': False
                })
                if (player == 2):
                    turns_hist[player][-1]['board_state_action'] = Gameplay.invert_board(new_boardState)

                boardState = new_boardState

            print("Game Over! ")
            if gm.move_limit_reached():
                print("It's a tie!!")
                for j in range(2):
                    turns_hist[j+1][-1]['reward'] += DRAW_REWARD
                    turns_hist[j+1][-1]['done'] = True
            else:
                print("Winner is: {}".format(gm.get_winner()))
                turns_hist[gm.get_winner()][-1]['reward'] += WIN_REWARD
                turns_hist[gm.get_winner()][-1]['done'] = True
                turns_hist[self.get_other_player(gm.get_winner())][-1]['reward'] -= WIN_REWARD
                turns_hist[self.get_other_player(gm.get_winner())][-1]['done'] = True

            for k, v in turns_hist.items():
                print("Reward sum for {}: {}".format(k, sum(list(map(lambda x: x['reward'], v)))))

            for k, v in turns_hist.items():
                for turn_hist in v:
                    self.agent.remember(turn_hist['board_state'], turn_hist['board_state_action'],
                                        turn_hist['reward'], turn_hist['next_board_state'],
                                        turn_hist['next_possible_board_states'], turn_hist['done'])

            if(len(self.agent.memory) > self.agent.replay_batch_size):
                self.agent.replay_memory()

        return self.agent

    @staticmethod
    def get_other_player(player):
        return 1 if player == 2 else 2
