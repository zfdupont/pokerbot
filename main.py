from models.player import Player
from agents.simple_agent import SimpleAgent
from agents.position_agent import PositionBasedAgent
from game.poker import PokerGame
from util.observer import ConsoleObserver

if __name__ == "__main__":
    players = [
        Player("Simple Player", 1000, SimpleAgent()),
        Player("Position Player", 1000, PositionBasedAgent()),
        Player("Simple Player 2", 1000, SimpleAgent()),
        Player("Position Player 2", 1000, PositionBasedAgent()),
    ]

    game = PokerGame(players, small_blind=1, observers=[ConsoleObserver()])
    game.play_hand()
