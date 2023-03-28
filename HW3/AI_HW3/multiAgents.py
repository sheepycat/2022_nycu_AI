from os import remove
from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
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
    Your minimax agent (Part 1)
    """
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        def minimax(depth,state,index):
            if(state.isLose() or state.isWin() or depth>self.depth): #stop and evalute when exceeding depth/ lose/ win
                return self.evaluationFunction(state)# evalute state
            
            legal_act = state.getLegalActions(index)# all possible action of current state & current ghost/pacman
            all_choice = []# save possible choice
            
            for act in legal_act : #each action
                next_state = state.getNextState(index,act)
                if((index+1) >= state.getNumAgents()): 
                    all_choice.append(minimax(depth+1,next_state,0)) #last ghost at current depth : repeat,append depth, index->pacman
                else:
                    all_choice.append(minimax(depth,next_state,index+1)) #else : repeat, next agent
                
            if(index==0):#pacman
                if(depth!=1):# not top 
                    choice = max(all_choice)#pacman choose max
                else:
                    best = max(all_choice)#pacman choose max
                    for c in range(len(all_choice)):
                        if(all_choice[c]==best):
                            return legal_act[c]#return responding action of best 
            elif(index > 0):# ghost
                choice = min(all_choice) # ghost choose min

            return choice # when not top return  choice after minimax
        return minimax(1,gameState,0) # start recursive
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        #raise NotImplementedError("To be implemented")
        def AlphaBeta(depth,state,index,a,b):
            # a = greatest lower bound
            # b = smallest upper bount

            if(state.isLose() or state.isWin() or depth>self.depth): #stop and evalute when exceeding depth/ lose/ win
                return self.evaluationFunction(state) # evalute state
            
            legal_act = state.getLegalActions(index)# all possible action of current state & current ghost/pacman
            all_choice = []# save possible choice
            
            for act in legal_act :  #each action
                next_state = state.getNextState(index,act)
                if((index+1) >= state.getNumAgents()):
                    v=AlphaBeta(depth+1,next_state,0,a,b)  #last ghost at current depth : repeat,append depth, index->pacman
                else:
                    v=AlphaBeta(depth,next_state,index+1,a,b) #else : repeat, next agent

                #pruning    
                if(index==0):#pacman
                    if(v>b): # if v > smallest upper bound 
                        return v
                    a = max(a,v)  # update new greatest lower bound if (a>v)
                if(index>0):#ghost
                    if(v<a): # if v < greatest lower bound 
                        return v
                    b = min(b,v) # update new smallest upper bound if (b>v)
                all_choice.append(v)# update choice of remaining v

            if(index==0):#pacman
                if(depth!=1): # not top
                    choice = max(all_choice)#pacman choose max
                else:
                    best = max(all_choice)#pacman choose max
                    for c in range(len(all_choice)):
                        if(all_choice[c]==best):
                            return legal_act[c]#return responding action of best 
            elif(index > 0):# ghost
                choice = min(all_choice)# ghost choose min
            return choice # when not top return choice
            
        return AlphaBeta(1,gameState,0,float('-Inf'),float('Inf')) # start recursive with initial a = -infinity, b = infinity
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        #raise NotImplementedError("To be implemented")
        def expectimax(depth,state,index):
            if(state.isLose() or state.isWin() or depth>self.depth):#stop and evalute when exceeding depth/ lose/ win
                return self.evaluationFunction(state)  # evalute state
            
            legal_act = state.getLegalActions(index)# all possible action of current state & current ghost/pacman
            all_choice = [] # save possible choice
            
            for act in legal_act :  #each action
                next_state = state.getNextState(index,act)
                if((index+1) >= state.getNumAgents()):
                    all_choice.append(expectimax(depth+1,next_state,0)) #last ghost at current depth : repeat,append depth, index->pacman
                else:
                    all_choice.append(expectimax(depth,next_state,index+1)) #else : repeat, next agent

                
            if(index==0): #pacman
                if(depth!=1):# not top
                    choice = max(all_choice)#pacman choose max
                else:
                    best = max(all_choice)#pacman choose max
                    for c in range(len(all_choice)):
                        if(all_choice[c]==best):
                            return legal_act[c] #return responding action of best 
            elif(index > 0): # ghost -> expect value
                choice = float(sum(all_choice)/len(all_choice)) # choose the choice with average value  

            return choice #when not top return choice
        return expectimax(1,gameState,0)
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
   
    minGhostDistance = min([manhattanDistance(Pos, state.getPosition()) for state in GhostStates])

    score = currentGameState.getScore()

    if(len(Food.asList())>0): #if food != 0
        NearestFoodDistance = min([manhattanDistance(Pos, food) for food in Food.asList()]) #nearest food
    else:
        NearestFoodDistance = 0

    if NearestFoodDistance>0 :  #if food != 0
            f_score = 10/NearestFoodDistance+5 # compute food score : higher if food is closer
    else :
        f_score = 0 
    if minGhostDistance>0 :
        if(sum(ScaredTimes)>3): # scaredtime>3 : higher possitive score to eat ghost
            g_score = 290/minGhostDistance
        elif(sum(ScaredTimes)>0): # scaredtime almost end : slow down to avoid killed by ghost, possitive score to eat ghost
            g_score = 150/minGhostDistance
        else : 
            g_score = -13/minGhostDistance # not scaredtime : minus more when the ghost is closer
    else:
        g_score = 0
    better = score + f_score + g_score # sum the base score, food score, ghost score

    return better
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
