from checkers import game
import numpy as np
from functools import reduce
from math import ceil

BOARD_SIZE = 8

class Gameplay(object):

    def calc_col(self, pos, row_width, row):
        return (2*(pos - 1 - row*row_width)) + ((row+1) % 2)

    def pos_coord(self, pos, gme):
        row = ceil(pos / gme.board.width) - 1
        return [row, self.calc_col(pos, gme.board.width, row)]

    def pos_from_coord(self, coord_row, coord_col, gme):
        return int(coord_row*gme.board.width + 1 + ((coord_col-((coord_row+1) % 2))/2))

    def move_from_pos(self, mv, gme):
        fr = mv[0]
        tt = mv[1]
        return [self.pos_coord(fr, gme), self.pos_coord(tt, gme)]

    def pos_move_from_coord(self,move_coord, gme):
        fr = move_coord[0]
        tt = move_coord[1]
        return [self.pos_from_coord(fr[0], fr[1], gme), self.pos_from_coord(tt[0], tt[1], gme)]

    def show_board(self, bd_state):
        rows = reduce((lambda x, y: "{} {}".format(x, y)), range(BOARD_SIZE))
        print(rows)
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

    def update_board_state(self,gme):
        board_tate = np.zeros((BOARD_SIZE,BOARD_SIZE))
        # W:1, B:2
        for piece in gme.board.pieces:
            if(not piece.captured):
                col = self.calc_col(piece.position, gme.board.width, piece.get_row())
                if(piece.player == 1): # W
                    if (piece.king):
                        board_tate[piece.get_row()][col] = 2
                    else:
                        board_tate[piece.get_row()][col] = 1
                else:
                    if (piece.king):
                        board_tate[piece.get_row()][col] = -2
                    else:
                        board_tate[piece.get_row()][col] = -1
        return board_tate

    def make_move(self, gme, move):
        gme.move(move)
        return self.update_board_state(gme)

    def run_game(self):
        gm = game.Game()
        boardState = self.update_board_state(gm)
        while(not gm.is_over()):
            self.show_board(boardState)
            print("Current Player: " + ("White" if gm.whose_turn() == 1 else "Black"))
            possible_moves_coord = list(map(lambda x: self.move_from_pos(x, gm), gm.get_possible_moves()))
            print("Valid Moves: ")
            for i in range(len(possible_moves_coord)):
                print(str(i + 1) + ": " + (str(possible_moves_coord[i])))
            move_idx = -1
            while move_idx not in range(len(possible_moves_coord)):
                usr_input = input("Pick a move: ")
                move_idx = -1 if (usr_input == '') else (int(usr_input)-1)
                if move_idx not in range(len(possible_moves_coord)):
                    print("Illegal move")

            move = gm.get_possible_moves()[move_idx]
            if (move in gm.board.get_possible_capture_moves()):
                print('capture move!')

            boardState = self.make_move(gm, move)

            if (gm.board.searcher.get_piece_by_position(move[1]).king):
                print('king move!')

        print("Game Over! ")
        if gm.move_limit_reached():
            print("It's a tie!!")
        else:
            print("Winner is: " + ("White" if gm.get_winner() == 1 else "Black"))

    def run_game_with_agent(self, agent):
        gm = game.Game()
        boardState = self.update_board_state(gm)
        while (not gm.is_over()):
            self.show_board(boardState)
            print("Current Player: " + ("White" if gm.whose_turn() == 1 else "Black"))
            possible_moves_coord = list(map(lambda x: self.move_from_pos(x, gm), gm.get_possible_moves()))
            print("Valid Moves: ")
            for i in range(len(possible_moves_coord)):
                print(str(i + 1) + ": " + (str(possible_moves_coord[i])))

            if(gm.whose_turn() == 1):
                move = -1
                while move not in range(len(possible_moves_coord)):
                    usr_input = input("Pick a move: ")
                    move = -1 if (usr_input == '') else (int(usr_input) - 1)
                    if move not in range(len(possible_moves_coord)):
                        print("Illegal move")
                print("Human picks {}: ".format(move))
            else:
                move = self.get_QAgent_move(agent, boardState, possible_moves_coord)
                print("Agent picks {}: ".format(move))

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

    def get_QAgent_move(self, agent, board_state, possible_moves):
        return agent.choose_action(board_state, possible_moves)
