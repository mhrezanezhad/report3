# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """

    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def min_value(self, gameState, advance_depth, Agent_index):# advace_depth = 0, Agent_index = 2

        Agent_index = Agent_index % gameState.getNumAgents()
        if gameState.isLose() or gameState.isWin():
            return betterEvaluationFunction(gameState)


        v = float('inf')
        for next_action in gameState.getLegalActions(Agent_index):
            if Agent_index < gameState.getNumAgents() - 1:
                v = min(v, self.min_value(gameState.generateSuccessor(Agent_index, next_action), advance_depth, Agent_index + 1))
            elif Agent_index == gameState.getNumAgents() - 1:
                v = min(v, self.max_value(gameState.generateSuccessor(Agent_index, next_action), advance_depth + 1, Agent_index + 1))
        return v

    def max_value(self, gameState, advance_depth, Agent_index):

        Agent_index = Agent_index % gameState.getNumAgents()

        if gameState.isLose() or gameState.isWin() or self.depth == advance_depth :
            return betterEvaluationFunction(gameState)

        legalActions = gameState.getLegalActions(Agent_index)
        nextAction = Directions.STOP
        v = float('-inf')

        for legalAction in legalActions:
             scoreOfAction = self.min_value(gameState.generateSuccessor(0, legalAction), advance_depth, Agent_index + 1)
             if scoreOfAction >= v:
                 v = scoreOfAction
                 nextAction = legalAction

        if advance_depth == 0:
            return nextAction
        else:
            return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.max_value(gameState, 0,0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def min_value(self, gameState, advance_depth, Agent_index):

        Agent_index = Agent_index % gameState.getNumAgents()
        if gameState.isLose() or gameState.isWin():
            return betterEvaluationFunction(gameState)


        v = float('inf')
        number_of_next_actions = len(gameState.getLegalActions(Agent_index))
        for next_action in gameState.getLegalActions(Agent_index):
            if Agent_index < gameState.getNumAgents() - 1:
                if v == float('inf'):
                    v = min(v, self.min_value(gameState.generateSuccessor(Agent_index, next_action), advance_depth, Agent_index + 1))
                else:
                    v += min(v, self.min_value(gameState.generateSuccessor(Agent_index, next_action), advance_depth, Agent_index + 1))

            elif Agent_index == gameState.getNumAgents() - 1:
                if v == float('inf'):
                    v = min(v, self.max_value(gameState.generateSuccessor(Agent_index, next_action), advance_depth + 1, Agent_index + 1))
                else:
                    v += min(v, self.max_value(gameState.generateSuccessor(Agent_index, next_action), advance_depth + 1, Agent_index + 1))
        return v / number_of_next_actions

    def max_value(self, gameState, advance_depth, Agent_index):

        Agent_index = Agent_index % gameState.getNumAgents()

        if gameState.isLose() or gameState.isWin() or self.depth == advance_depth :
            return betterEvaluationFunction(gameState)

        legalActions = gameState.getLegalActions(Agent_index)
        nextAction = Directions.STOP
        v = float('-inf')

        for legalAction in legalActions:
             scoreOfAction = self.min_value(gameState.generateSuccessor(0, legalAction), advance_depth, Agent_index + 1)
             if scoreOfAction >= v:
                 v = scoreOfAction
                 nextAction = legalAction
        if advance_depth == 0:
            return nextAction
        else:
            return v

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.max_value(gameState, 0, 0)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    if currentGameState.isWin():
        return 100000 + currentGameState.getScore()
    if currentGameState.isLose():
        return -100000 + currentGameState.getScore()



    result = currentGameState.getScore() + 8 * 1 / BFS_for_evaluation_Fucntion(currentGameState)

    return result


def nextLegalPositions(gameState, xy):
    nextPositions = []
    if not gameState.hasWall(xy[0] + 1, xy[1]):
        nextPositions.append([xy[0] + 1, xy[1]])

    if not gameState.hasWall(xy[0] - 1, xy[1]):
        nextPositions.append([xy[0] - 1, xy[1]])

    if not gameState.hasWall(xy[0], xy[1] + 1):
        nextPositions.append([xy[0], xy[1] + 1])

    if not gameState.hasWall(xy[0], xy[1] - 1):
        nextPositions.append([xy[0], xy[1] - 1])

    return nextPositions

def BFS_for_evaluation_Fucntion(gameState: GameState):

    li = [gameState.getPacmanPosition()]
    visited = [gameState.getPacmanPosition()]
    depth = 0
    while len(li):
        depth += 1
        lenth = len(li)
        for i in range(lenth):
            nextPositions = nextLegalPositions(gameState, li[i])
            for j in nextPositions:
                if gameState.hasFood(j[0], j[1]):
                    return depth
                elif not(j in visited):
                    li.append(j)
                    visited.append(j)


#Abbreviation
better = betterEvaluationFunction
