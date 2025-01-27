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
       

        # Calculate the distance to the nearest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 1

        # Calculate the distance to the nearest ghost
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else 1

        # Calculate scared ghost impact if power pellet is nearby
        scaredGhostImpact = 0
        for ghost, scaredTime in zip(newGhostStates, [ghostState.scaredTimer for ghostState in newGhostStates]):
            if scaredTime > 0:  # Ghost is scared
                ghostDistance = manhattanDistance(newPos, ghost.getPosition())
                scaredGhostImpact += 5.0 / (ghostDistance + 1)  # Prioritize eating scared ghosts

        # Base score
        score = successorGameState.getScore()

        # Encourage getting closer to food and consider scared ghosts
        score += 9.0 / closestFoodDistance  # Higher score for being closer to food
        score += scaredGhostImpact  # Reward for being closer to scared ghosts

        # Penalize getting closer to non-scared ghosts
        if closestGhostDistance == 0:  # Pac-Man reached a ghost
            score -= 100  # High penalty for being on the same position as a ghost
        else:
            score -= 2.0 / closestGhostDistance  
            
        return score
 
        

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
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            # If we reach the maximum depth, win or lose state, return evaluation
            if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0:  # Pac-Man's turn (maximizer)
                return max_value(agentIndex, depth, gameState)
            else:  # Ghosts' turn (minimizer)
                return min_value(agentIndex, depth, gameState)

        def max_value(agentIndex, depth, gameState):
            v = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minimax((agentIndex + 1) % gameState.getNumAgents(), depth + 1, successorState))
            return v

        def min_value(agentIndex, depth, gameState):
            v = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(gameState)
            for action in legalActions:
                successorState = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax((agentIndex + 1) % gameState.getNumAgents(), depth + 1, successorState))
            return v

        # Choose the best action for Pac-Man (agentIndex 0)
        legalActions = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(1, 1, successorState)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
    
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Start the recursive function for alpha-beta pruning
        _, best_action = self.alphaBeta(gameState, 0, 0, float('-inf'), float('inf'))
        return best_action
    
    def alphaBeta(self, gameState, depth, agentIndex, alpha, beta):
         # If the current state is terminal, return the utility (game state evaluation)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Determine if we are at a max (Pacman) or min (ghost) node
        if agentIndex == 0:  # Pacman's turn (maximize)
            return self.maxValue(gameState, depth, alpha, beta)
        else:  # Ghost's turn (minimize)
            return self.minValue(gameState, depth, agentIndex, alpha, beta)

    def maxValue(self, gameState, depth, alpha, beta):
        max_score = float('-inf')
        best_action = None
        legalMoves = gameState.getLegalActions(0)

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score, _ = self.alphaBeta(successor, depth, 1, alpha, beta)
            
            if score > max_score:
                max_score = score
                best_action = action
            
            alpha = max(alpha, max_score)
            
            if max_score > beta:  # Prune here if max_score is strictly greater than beta
                break

        return max_score, best_action

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        min_score = float('inf')
        best_action = None
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1

        if nextAgent >= gameState.getNumAgents():  # Loop back to Pacman
            nextAgent = 0
            depth += 1

        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.alphaBeta(successor, depth, nextAgent, alpha, beta)
            
            if score < min_score:
                min_score = score
                best_action = action
            
            beta = min(beta, min_score)

            if min_score < alpha:  # Prune here if min_score is strictly less than alpha
                break

        return min_score, best_action

    
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Start the recursive function for expectimax
        _, best_action = self.expectimax(gameState, 0, 0)
        return best_action

    def expectimax(self, gameState, depth, agentIndex):
        # If the current state is terminal, return the utility (game state evaluation)
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None

        # Determine if we are at a max (Pacman) or expect (ghost) node
        if agentIndex == 0:  # Pacman's turn (maximize)
            return self.maxValue(gameState, depth)
        else:  # Ghost's turn (expect)
            return self.expectValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth):
        max_score = float('-inf')
        best_action = None
        legalMoves = gameState.getLegalActions(0)

        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score, _ = self.expectimax(successor, depth, 1)
            
            if score > max_score:
                max_score = score
                best_action = action

        return max_score, best_action

    def expectValue(self, gameState, depth, agentIndex):
        expect_score = 0
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = agentIndex + 1

        if nextAgent >= gameState.getNumAgents():  # Loop back to Pacman
            nextAgent = 0
            depth += 1

        for action in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, action)
            score, _ = self.expectimax(successor, depth, nextAgent)
            expect_score += score

        if legalMoves:  # Prevent division by zero
            expect_score/= len(legalMoves)
        
        return expect_score,None

         
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
Function description: 
- It gives points for getting close to food, so Pacman wants to eat it fast.
- It takes away points if Pacman is too close to "scary" ghosts, so he stays safe.
- It gives extra points for going after scared ghosts, so Pacman can catch them.
- It also gives points for getting close to power capsules, which turn ghosts blue.
- Finally, it adds a small penalty for any food left, so Pacman keeps eating until hes done.
    """
    "*** YOUR CODE HERE ***"
    # Initialize the score with the current state's score as a base
    score = currentGameState.getScore()
    
    # Pacman position
    pacmanPos = currentGameState.getPacmanPosition()
    
    # Food positions and distances
    food = currentGameState.getFood()
    foodList = food.asList()
    if foodList:
        closestFoodDist = min(manhattanDistance(pacmanPos, foodPos) for foodPos in foodList)
        score += 10.0 / closestFoodDist  # Reward being close to food
    
    # Ghost positions, scared states, and distances
    for ghostState in currentGameState.getGhostStates():
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)
        
        if ghostState.scaredTimer > 0:
            # Encourage moving towards scared ghosts to eat them
            score += 200.0 / ghostDist if ghostDist > 0 else 200
        else:
            # Penalize proximity to active ghosts to avoid being eaten
            score -= 10.0 / ghostDist if ghostDist > 0 else 500
    
    # Remaining capsules (which allow Pacman to eat ghosts)
    capsules = currentGameState.getCapsules()
    if capsules:
        closestCapsuleDist = min(manhattanDistance(pacmanPos, capPos) for capPos in capsules)
        score += 50.0 / closestCapsuleDist  # Encourage getting closer to capsules
    
    # Penalize based on the amount of food left, encouraging Pacman to finish the level
    score -= 4 * len(foodList)
    
    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
