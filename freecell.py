#!/usr/bin/env python3

import time
import os
import sys
import copy
from queue import LifoQueue, PriorityQueue
import ctypes
import ctypes.wintypes
from multiprocessing import popen_spawn_win32
from subprocess import check_output
from collections import deque
import operator

MAX_DEPTH = 6
global visited
visited = set()
global max_foundation_cards
max_foundation_cards=[0]

card_color = {
    "B": [0, 1],
    "R": [2, 3]
}

suit = {
    "S": 0,
    "C": 1,
    "H": 2,
    "D": 3
}

def find_color(card):
    for i in card_color.keys():
        if find_suit(card) in card_color[i]:
            return i
    return None

def find_suit(card):
    letter = card[0]

    if letter == 'H' or letter == 'h':
        return suit["H"]
    elif letter == 'S' or letter == 's':
        return suit["S"]
    elif letter == 'D' or letter == 'd':
        return suit["D"]
    elif letter == 'C' or letter == 'c':
        return suit["C"]
    return None

def find_value(card):
    value = int(card[1:])

    if type(value) != int:
        return None
    return value

def tableau_move_valid(card1, card2):
    if find_color(card1) != find_color(card2):
        valueDiff = find_value(card1) - find_value(card2)
        if valueDiff == -1:
            return -1 # move card1 below card2
        elif valueDiff == 1:
            return 1 # move card2 below card1
        else:
            return 0 # can't move

# Check whether a card can be put in the foundation
def foundation_move_valid(card_value, foundation):
    if not foundation:
        foundation_val = -1
    else:
        foundation_val= find_value(foundation[-1]) #last foundation card val
    return card_value == foundation_val + 1

class TreeNode(object):
    def __init__(self, board, parent, move, num_moves):
        self.board = board
        self.move = move
        self.num_moves = num_moves
        self.parent = parent
        self.children = []
        self.score = 0
        length=0
        for foundation in self.board["foundation"].values():
            length+=len(foundation)
        self.foundation_cards = length

    #The heuristic function
    def heuristic(self):
        copy_board = copy.deepcopy(self.board)
       
        count = 0
        found_multiplier = False
        for foundation_id in copy_board["foundation"]:
            if copy_board['foundation'][foundation_id] != []:
                found_multiplier = True
            curr_foundation_list = copy_board['foundation'][foundation_id]
            for tableau_id in copy_board["tableau"]:
                tableau_len = len(copy_board['tableau'][tableau_id])
                for id, card in enumerate(copy_board["tableau"][tableau_id]):
                    if card:
                        value = find_value(card)
                        if foundation_move_valid(value, curr_foundation_list):
                            cards_left = tableau_len - 1 - id
                            count+=cards_left
        
        freecell_multiplier = True
        for freecell in copy_board["freecell"]:
            if (freecell):
                freecell_multiplier = False

        if (freecell_multiplier or found_multiplier):
            return (2*count)
        else:
            return (count)

    def possible_moves(self):
        moves = []

        # STEP 1 loop for the freecells
        for i in range(4):
            if self.board["freecell"][i]:
                suit = find_suit(self.board["freecell"][i])
                card_value = find_value(self.board["freecell"][i])
                foundation = self.board["foundation"][suit]
                # if card can be moved to foundation, do it
                if foundation_move_valid(card_value, foundation):
                    new_board, move = self.freecell_to_foundation(i, suit)
                    moves.append([new_board, move])
                    continue

                # move card from freecell to tableau
                for j in range(1, 9):
                    curr_tableau = self.board["tableau"][j]

                    # if curr_tableau empty, then move the freecell card to it
                    # else, check if the card can be moved to tableau
                    if not curr_tableau:
                        new_board, move = self.freecell_to_empty_tableau(i, j)
                        moves.append([new_board, move])
                        continue
                    elif tableau_move_valid(curr_tableau[-1], self.board["freecell"][i]) == 1:
                        new_board, move = self.freecell_to_tableau(i,j)
                        moves.append([new_board, move])
                        continue
            elif self.board["freecell"][i] is None:
                # move a card from tableau to freecell
                for j in range(1,9):
                    if not self.board["tableau"][j]:
                        continue
                    else:
                        new_board, move = self.tableau_to_freecell(i, j)
                        moves.append([new_board, move])
            
        # STEP 2 loop through the tableaus
        for i in range(1,9):
            # Tableau to tableau
            for j in range(i+1, 9):
                src_id = 0
                dest_id = 0
                if self.board["tableau"][i] == [] and self.board["tableau"][j] == []:
                    continue
                # fill empty tableaus (source)
                elif self.board["tableau"][i] == [] and self.board["tableau"][j] != []:
                    if len(self.board["tableau"][j]) == 1:
                        continue
                    else:
                        src_id = j
                        dest_id = i
                elif self.board["tableau"][i] != [] and self.board["tableau"][j] == []:
                    if len(self.board["tableau"][i]) == 1:
                        continue
                    else:
                        dest_id = j
                        src_id = i
                elif self.board["tableau"][i] != [] and self.board["tableau"][j] != []:
                    card1 = self.board["tableau"][i][-1]
                    card2 = self.board["tableau"][j][-1]
                    if tableau_move_valid(card1, card2) == 1:
                        src_id = j
                        dest_id = i
                    elif tableau_move_valid(card1, card2) == -1:
                        dest_id = j
                        src_id = i
                    else:
                        continue
                new_board, move = self.tableau_to_tableau(dest_id, src_id)
                moves.append([new_board, move])
                
            # Tableau to foundation
            if self.board["tableau"][i]:
                tableau_last= self.board["tableau"][i][-1]
                suit = find_suit(tableau_last)
                card_value = find_value(tableau_last)
                # Move from tableau to foundation
                if foundation_move_valid(card_value, self.board["foundation"][suit]):
                    new_board, move = self.tableau_to_foundation(i, suit)
                    moves.append([new_board, move])
                    continue
        return moves

    def freecell_to_foundation(self, freecell_index, foundation):
        copy_board = copy.deepcopy(self.board)
        card = copy_board["freecell"][freecell_index]
        copy_board["freecell"][freecell_index] = None
        copy_board["foundation"][foundation].append(card)
        return copy_board, ("foundation", card)

    def freecell_to_empty_tableau(self, freecell_index, tableau_id):
        copy_board = copy.deepcopy(self.board)
        card = copy_board["freecell"][freecell_index]
        copy_board["freecell"][freecell_index] = None
        copy_board["tableau"][tableau_id].append(card)
        return copy_board, ("source", card)

    def freecell_to_tableau(self, freecell_index, tableau_id):
        copy_board = copy.deepcopy(self.board)
        card = copy_board["freecell"][freecell_index]
        if copy_board["tableau"][tableau_id] is []:
            copy_board["freecell"][freecell_index] = None
            copy_board["tableau"][tableau_id].append(card)
            return copy_board, ("source", card)
        else:
            tableau_last = copy_board["tableau"][tableau_id][-1]
            if tableau_move_valid(tableau_last, card) == 1:
                copy_board["freecell"][freecell_index] = None
                copy_board["tableau"][tableau_id].append(card)
                return copy_board, ("tableau", card, tableau_last)
        return None

    def tableau_to_freecell(self, freecell_index, tableau_id):
        copy_board = copy.deepcopy(self.board)
        card = copy_board["tableau"][tableau_id].pop()
        copy_board["freecell"][freecell_index] = card
        return copy_board, ("freecell", card)

    def tableau_to_foundation(self, tableau_id, foundation):
        copy_board = copy.deepcopy(self.board)
        card = copy_board["tableau"][tableau_id].pop()
        copy_board["foundation"][foundation].append(card)
        return copy_board, ("foundation", card)

    def tableau_to_tableau(self, src_tableau, dest_tableau):
        copy_board = copy.deepcopy(self.board)
        last_dest_tableau = copy_board["tableau"][dest_tableau].pop()
        if copy_board["tableau"][src_tableau]:
            last_src_tableau = copy_board["tableau"][src_tableau][-1]
            copy_board["tableau"][src_tableau].append(last_dest_tableau)
            move = ('tableau', last_dest_tableau, last_src_tableau)
        else:
            copy_board["tableau"][src_tableau].append(last_dest_tableau)
            move = ('source', last_dest_tableau)
        return copy_board, move
    
    def find_children(self):
        # this function adds children and sets the heuristic score
        moves = self.possible_moves()
        global max_foundation_cards
        for state in moves:
            num_moves = self.num_moves + 1
            child = TreeNode(state[0], self, state[1], num_moves=num_moves)
            child.score = child.heuristic()
            
            if child.foundation_cards>max_foundation_cards[-1]:
                max_foundation_cards.append( child.foundation_cards )
                child.score-=1000
                self.children.append(child)
            elif child.foundation_cards==max_foundation_cards[-1]:
                self.children.append(child)

    def check_win(self):
        for i in range(4):
            if len(self.board["foundation"][i]) < 13:
                return False
        return True

    # Solution steps:
    def get_solution_path(self):
        path = []
        temp_node = copy.copy(self)
        while temp_node:
            if temp_node.move:
                path.append(temp_node.move)
            temp_node = temp_node.parent
        return path

def HSD(init_board, N):
    root = TreeNode(init_board, None, None, 0)
    T = {root: root.score} # T <- initial state
    global visited
    global max_foundation_cards

    s=(root, root.score)
    no_solution = False
    dead_end = False
    while ( T ): # while T not empty do
        s = T.popitem()
        while (s[0].foundation_cards < max_foundation_cards[-1] and dead_end == False):
            print('popping')
            print(s[0].foundation_cards)
            print(max_foundation_cards[-1])
            if T:
                s = T.popitem() # s<-remove best state in T
            else:
                print('NO SOLUTION FOUND')
                no_solution = True
                break

        visited.add(str(s[0].board))

        if no_solution:
            break
        
        U = DFS(s[0], MAX_DEPTH) # U <- all possible states exactly k moves away from s, DFS
    
        if not U:
            print('DEAD END')
            dead_end = True
            max_foundation_cards=[s[0].parent.foundation_cards]
            U = DFS(s[0].parent, 6)

        T=merge(T, U)   # merge U into T

        print(len(T))
        key=list(T)[-1]
        score = list(T.values())[-1]
        print('BOARD '+str(key.board)+' SCORE '+str(score)+' FOUNDATION '+str(key.foundation_cards)+' MAX_FOUND '+str(max_foundation_cards))

        if len(visited) >= N: # if size of T >= N then clear
            visited.clear()
        
        for i in T.keys():
            if i.check_win():   # if goal in T, return node
                return i

def DFS(s, k):
    global visited

    if k == 0:
        return [s]
    
    if s.check_win():
        return [s]

    s.find_children()
    U = []
    for child in s.children:
        if str(child.board) not in visited and child.foundation_cards >= max_foundation_cards[-1]:
            visited.add(str(child.board)) 
            U.extend(DFS(child, k-1))
    return U

def merge(T, U):
    merged_states = list(T)
    for new_state in U:
        if len(merged_states) != 0:
            for id, score in enumerate(T.values()):
                if score == new_state.score:
                    merged_states[id] = new_state
                elif new_state.score >= score:
                    continue
        else:
            merged_states.append(new_state)
        if new_state not in merged_states:
            merged_states.append(new_state)

    mydict = {}
    for i in merged_states:
        mydict[i] = i.score
    sorted_dict = dict( sorted(mydict.items(), key=operator.itemgetter(1), reverse=True) )
    return sorted_dict


# Load data
NumToCard = {}
Cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
Suits = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
# cards you can place in the columns
Move_Values = {'A':'2', '2':'3', '3':'4', '4':'5', '5':'6', '6':'7', '7':'8', '8':'9', '9':'10', '10':'J', 'J':'Q', 'Q':'K', 'K':'empty'}

class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit
        if suit == "Diamonds" or suit == 'Hearts':
            self.color = 'Red'
        else:
            self.color = 'Black'

class Column:
    def __init__(self, cards):
        # create new column of cards
        self.cards = cards
    def pop(self):
        # pop last card in the column
        self.cards = self.cards[:-1]
    def append(self, new_card):
        pile = []
        for card in self.cards:
            pile.append(card)
        pile.append(new_card)
        self.cards = pile
    def end_card(self):
        # return the last card in the column
        return self.cards[-1]
    def total_cards(self):
        return len(self.cards)
    def empty_col(self):
        if len(self.cards) == 0:
            return True
        else:
            return False
    def cascading_amt(self):
        # given column return the number of cards in the cascade
        count = 0
        if self.total_cards() > 1:
            current_card = self.total_cards()-1
            checking_card = self.total_cards()-2
            while checking_card >= 0: # while there are still cards
                if (self.cards[checking_card].value == Move_Values[self.cards[current_card].value] and self.cards[checking_card].color != self.cards[current_card].color):
                    count += 1
                    current_card -= 1
                    checking_card -= 1
                else:
                    break
        return count
    def cascade(self):
        return self.cascading_amt() > 0

def get_pid():
    l = check_output('tasklist /fi "Imagename eq freecell.exe"').split()
    return int(l[14])

def setup():
    index = 0
    for card in Cards:
        for suit in Suits:
            hex_val = str(hex(index))
            NumToCard[hex_val] = [card, suit]
            index += 1
    kernel32 = ctypes.windll.kernel32
    # have to go open freecell and check pid !!
    freecell_pid = get_pid()
    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = ctypes.wintypes.DWORD,ctypes.wintypes.BOOL,ctypes.wintypes.DWORD
    OpenProcess.restype = ctypes.wintypes.HANDLE
    ReadProcessMemory = kernel32.ReadProcessMemory
    ReadProcessMemory.argtypes = ctypes.wintypes.HANDLE, ctypes.wintypes.LPCVOID,ctypes.wintypes.LPCVOID,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t)
    ReadProcessMemory.restype = ctypes.wintypes.BOOL
    PROCESS_VM_READ = 0x0010
    PROCESS_VM_WRITE = 0x0020
    addr = 0x01008B00
    data = ctypes.c_ulonglong()
    bytesRead = ctypes.c_ulonglong()
    hChrome = OpenProcess(PROCESS_VM_READ | PROCESS_VM_WRITE, False, freecell_pid)
    numbers = []
    while addr < 0x01008D64:
        addr += 4
        #print('new addr', hex(addr))
        data = ctypes.c_ulonglong()
        bytesRead = ctypes.c_ulonglong()
        result = ReadProcessMemory(hChrome, addr, ctypes.byref(data), ctypes.sizeof(data), ctypes.byref(bytesRead))
        error = ctypes.get_last_error()
        keep = False
        num = ''
        for val in str(data):
            if val == '(':
                keep = True
                continue
            if keep and val != ')':
                num += val
        if str(hex(int(num)))[-2:] == 'ff':
            continue
        if len(str(hex(int(num)))) > 3:
            append_val = '0x' + str(hex(int(num)))[-2:]
        else:
            append_val = str(hex(int(num)))
        if append_val[-2] == '0':
            append_val = append_val[:-2] + append_val[-1]
        numbers.append(append_val)
        if error != 0:
            print("last error", error)
   # print('numbers:', numbers)
    return numbers

def make_board(cards):
    col1 = []
    col2 = []
    col3 = []
    col4 = []
    col5 = []
    col6 = []
    col7 = []
    col8 = []

    for i in range(0, 7):
        col1.append(cards[i])
    for i in range(7, 14):
        col2.append(cards[i])
    for i in range(14, 21):
        col3.append(cards[i])
    for i in range(21, 28):
        col4.append(cards[i])
    for i in range(28, 34):
        col5.append(cards[i])
    for i in range(34, 40):
        col6.append(cards[i])
    for i in range(40, 46):
        col7.append(cards[i])
    for i in range(46, 52):
        col8.append(cards[i])

    board = [col1, col2, col3, col4, col5, col6, col7, col8]
    for i in range(len(board)):
        board[i] = Column(board[i])

    for i in range(len(board)):
        for j in range(len(board[i].cards)):
            card = Card(NumToCard[board[i].cards[j]][0], NumToCard[board[i].cards[j]][1])
            board[i].cards[j] = card
    return board

def change_board(board):
    new_board = []
    Card_values = {"A":"0", "2": "1", "3":"2", "4": "3", "5":"4", "6": "5", "7":"6", "8": "7", "9":"8", "10": "9", "J":"10", "Q": "11", "K":"12"}
    Card_suits = {"Diamonds":"D", "Hearts":"H", "Clubs":"C", "Spades":"S"}
    for column in board:
        string = ""
        for card in column.cards:
            string += Card_suits[card.suit] + Card_values[card.value] + " "
        new_board.append(string[:-1])
    return new_board

def initialize_board(data):
    board = {}

    # tableau id from 1 to 8
    board["tableau"] = {}
    id=1
    for line in data:
        item = line.split(' ')
        board["tableau"][id] = item
        id+=1

    # Initialize freecells
    board["freecell"] = []
    for i in range(4):
        board["freecell"].append(None)

    # Initialize the foundations
    board["foundation"] = {}
    for j in range(4):
        board["foundation"][j] = []

    return board

def write_solution_to_file(file, solution_steps, solution_moves):
    with open(file, "w") as file:
        file.write(f"{solution_steps}\n")
        for move in solution_moves[::-1]:
            if len(move) == 3:
                move_card, card1, card2 = move
                file.write("{} {} {} \n".format(move_card, card1, card2))
            elif len(move) == 2:
                move_card, card = move
                file.write("{} {} \n".format(move_card, card))

if __name__ == '__main__':
    start = time.time()

    if len(sys.argv) == 2:
        output_file = sys.argv[1]
    else:
        print(f'Usage: {sys.argv[0]} <output file name>')
        sys.exit()

    # Initialize the queue
    q = LifoQueue()

    # Read the data and initialize the problem
    cards = setup()
    orig_board = make_board(cards)
    data = change_board(orig_board)
    init_board = initialize_board(data)

    print("STARTING BOARD:\n" )

    #solution_node = search(q, init_board)
    solution_node = HSD(init_board, 200000)

    # If the solution has been found, then write to a file the steps of the solution
    if solution_node:
        print("SOLUTION FOUND!\n")
        print("Number of moves: {}".format(solution_node.num_moves))
        print("Time taken: ", time.time() - start)

        num_solution_steps = solution_node.num_moves
        solution_path = solution_node.get_solution_path()

        solution_path_format = []
        for i in range(num_solution_steps):
            if len(solution_path[i]) == 3:
                solution_path_format.append((solution_path[i][0], solution_path[i][1][0]+str(int(solution_path[i][1][1:])+1), solution_path[i][2][0]+str(int(solution_path[i][2][1:])+1)) )
            elif len(solution_path[i]) == 2:
                solution_path_format.append( (solution_path[i][0], solution_path[i][1][0]+str(int(solution_path[i][1][1:])+1)) )

        # Create output file for solution
        if len(sys.argv) == 2:
            try:
                write_solution_to_file(output_file, num_solution_steps, solution_path_format)
            except FileNotFoundError:
                write_solution_to_file(output_file, num_solution_steps, solution_path_format)
        else:
            write_solution_to_file(output_file, num_solution_steps, solution_path_format)
    else:
        print("Time : ", time.time() - start)
        print("NO SOLUTION FOUND")
        sys.exit()