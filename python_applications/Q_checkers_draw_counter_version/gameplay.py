from checkers import game
import numpy as np
from functools import reduce
from math import ceil
import random
from copy import deepcopy
import gc

BOARD_SIZE = 8
HUMAN = 'human'
AGENT = 'agent'


class Gameplay(object):

    @staticmethod
    def calc_col(pos, row_width, row):
        return (2*(pos - 1 - row*row_width)) + ((row+1) % 2)

    @staticmethod
    def pos_coord(pos, gme):
        row = ceil(pos / gme.board.width) - 1
        return [row, Gameplay.calc_col(pos, gme.board.width, row)]

    @staticmethod
    def pos_from_coord(coord_row, coord_col, gme):
        return int(coord_row*gme.board.width + 1 + ((coord_col-((coord_row+1) % 2))/2))

    @staticmethod
    def move_from_pos(mv, gme):
        fr = mv[0]
        tt = mv[1]
        return [Gameplay.pos_coord(fr, gme), Gameplay.pos_coord(tt, gme)]

    @staticmethod
    def pos_move_from_coord(move_coord, gme):
        fr = move_coord[0]
        tt = move_coord[1]
        return [Gameplay.pos_from_coord(fr[0], fr[1], gme), Gameplay.pos_from_coord(tt[0], tt[1], gme)]

    @staticmethod
    def show_board(bd_state):
        rows = reduce((lambda x, y: "{} {}".format(x, y)), range(BOARD_SIZE))
        print(rows)
        #print(bd_state)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if (bd_state[row][col] == 0):
                    print("+ ", end="")
                elif (bd_state[row][col] == 1):
                    print("W ", end="")
                elif (bd_state[row][col] == -1):
                    print("B ", end="")
                elif (bd_state[row][col] == 2):
                    print("\033[4mW\033[0m ", end="")
                elif (bd_state[row][col] == -2):
                    print("\033[4mB\033[0m ", end="")
            print(str(row))

    @staticmethod
    def board_state_from_board(board):
        board_tate = np.zeros((BOARD_SIZE,BOARD_SIZE))
        # W:1, B:2
        for piece in board.pieces:
            if (not piece.captured):
                col = Gameplay.calc_col(piece.position, board.width, piece.get_row())
                if (piece.player == 1): # W
                    if piece.king:
                        board_tate[piece.get_row()][col] = 2
                    else:
                        board_tate[piece.get_row()][col] = 1
                else:
                    if piece.king:
                        board_tate[piece.get_row()][col] = -2
                    else:
                        board_tate[piece.get_row()][col] = -1
        return board_tate

    @staticmethod
    def board_state_from_move(board, move):
        return Gameplay.board_state_from_board(board.create_new_board_from_move(move))

    @staticmethod
    def board_states_from_possible_moves(board):
        return np.array(list(map(lambda x: Gameplay.board_state_from_move(board, x), board.get_possible_moves())))

    @staticmethod
    def make_move(gme, move):
        gme.move(move)
        return Gameplay.board_state_from_board(gme.board)

    @staticmethod
    def invert_board(board_state):
        return np.array(list(map(lambda x: x[::-1]*(-1), board_state)))[::-1]

    def run_game(self):
        gm = game.Game()
        boardState = self.board_state_from_board(gm.board)
        while(not gm.is_over()):
            self.show_board(boardState)
            print("Current Player: " + ("White" if gm.whose_turn() == 1 else "Black"))
            possible_moves_coord = list(map(lambda x: self.move_from_pos(x, gm), gm.get_possible_moves()))
            print("Valid Moves: ")
            for i in range(len(possible_moves_coord)):
                print(str(i + 1) + ": " + (str(possible_moves_coord[i])))

            for bd_st in self.board_states_from_possible_moves(gm.board):
                self.show_board(bd_st)

            move_idx = -1
            while move_idx not in range(len(possible_moves_coord)):
                usr_input = input("Pick a move: ")
                move_idx = -1 if (usr_input == '') else (int(usr_input)-1)
                if move_idx not in range(len(possible_moves_coord)):
                    print("Illegal move")

            move = gm.get_possible_moves()[move_idx]
            if (move in gm.board.get_possible_capture_moves()):
                print('capture move!')

            piece_was_king = gm.board.searcher.get_piece_by_position(move[0]).king
            boardState = self.make_move(gm, move)

            if (not piece_was_king) and gm.board.searcher.get_piece_by_position(move[1]).king:
                print('king move!')

        print("Game Over! ")
        if gm.move_limit_reached():
            print("It's a tie!!")
        else:
            print("Winner is: " + ("White" if gm.get_winner() == 1 else "Black"))

    def run_game_with_agent(self, agent):
        players = [HUMAN, AGENT]
        gm = game.Game()
        boardState = self.board_state_from_board(gm.board)
        random.shuffle(players)
        while (not gm.is_over()):
            self.show_board(boardState)
            print("Current Player: " + ("White" if gm.whose_turn() == 1 else "Black"))
            possible_moves_coord = list(map(lambda x: self.move_from_pos(x, gm), gm.get_possible_moves()))
            print("Valid Moves: ")
            for i in range(len(possible_moves_coord)):
                print(str(i + 1) + ": " + (str(possible_moves_coord[i])))

            if(players[gm.whose_turn()-1] == HUMAN):
                move = -1
                while move not in range(len(possible_moves_coord)):
                    usr_input = input("Pick a move: ")
                    move = -1 if (usr_input == '') else (int(usr_input) - 1)
                    if move not in range(len(possible_moves_coord)):
                        print("Illegal move")
                print("Human picks {}: ".format(move))
            else:
                move, q_val = self.get_QAgent_move_pp(agent, gm.board)
                print("Agent {} picks {}: ".format(agent.name, move+1))
                print("Agent {} Q-value {}: ".format(agent.name, q_val))

            boardState = self.make_move(gm, gm.get_possible_moves()[move])

        print("Game Over! ")
        if gm.move_limit_reached():
            print("It's a tie!!")
        else:
            print("Winner is: " + ("White" if gm.get_winner() == 1 else "Black"))

    def get_human_player_move(self, possible_moves):
        move = -1
        while move not in range(len(possible_moves)):
            usr_input = input("Pick a move: ")
            move = -1 if (usr_input == '') else (int(usr_input) - 1)
            if move not in range(len(possible_moves)):
                print("Illegal move")
        return move

    @staticmethod
    def get_QAgent_move_pp(agent, gme):
        return agent.choose_action_pp(gme)
