from checkers import game
import numpy as np
from functools import reduce
from math import ceil

BOARD_SIZE = 8


def calc_col(pos, row_width, row):
    return (2*(pos - 1 - row*row_width)) + ((row+1) % 2)


def pos_coord(pos, gme):
    row = ceil(pos / gme.board.width) - 1
    return [row, calc_col(pos, gme.board.width, row)]


def pos_from_coord(coord_row, coord_col, gme):
    return int(coord_row*gme.board.width + 1 + ((coord_col-((coord_row+1) % 2))/2))


def move_from_pos(mv, gme):
    fr = mv[0]
    tt = mv[1]
    return [pos_coord(fr, gme), pos_coord(tt, gme)]


def pos_move_from_coord(move_coord, gme):
    fr = move_coord[0]
    tt = move_coord[1]
    return [pos_from_coord(fr[0], fr[1], gme), pos_from_coord(tt[0], tt[1], gme)]


def show_board(bd_state):
    rows = reduce((lambda x, y: "{} {}".format(x, y)), range(BOARD_SIZE))
    print(rows)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if (bd_state[row][col] == 0):
                print("+ ")
            elif (bd_state[row][col] == 1):
                print("W ")
            elif (bd_state[row][col] == -1):
                print("B ")
            elif (bd_state[row][col] == 2):
                print("\033[4mW[0m")
            elif (bd_state[row][col] == -2):
                print("\033[4mB\033[0m ")
        print(str(row))


def update_board_state(gme):
    board_tate = np.zeros((BOARD_SIZE,BOARD_SIZE))
    # W:1, B:2
    for piece in gme.board.pieces:
        if(not piece.captured):
            col = calc_col(piece.position, gme.board.width, piece.get_row())
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


def make_move(gme, move):
    gme.move(move)
    return update_board_state(gme)


def run_game():
    gm = game.Game()
    boardState = update_board_state(gm)
    while(not gm.is_over()):
        show_board(boardState)
        print("Current Player: " + ("White" if gm.whose_turn() == 1 else "Black"))
        possible_moves_coord = list(map(lambda x: move_from_pos(x, gm), gm.get_possible_moves()))
        print("Valid Moves: ")
        for i in range(len(possible_moves_coord)):
            print(str(i + 1) + ": " + (str(possible_moves_coord[i])))
        move = -1
        while move not in range(len(possible_moves_coord)):
            usr_input = input("Pick a move: ")
            move = -1 if (usr_input == '') else (int(usr_input)-1)
            if move not in range(len(possible_moves_coord)):
                print("Illegal move")

        boardState = make_move(gm, gm.get_possible_moves()[move])

    print("Game Over! ")
    if gm.move_limit_reached():
        print("It's a tie!!")
    else:
        print("Winner is: " + ("White" if gm.get_winner() == 1 else "Black"))



#run_game()

