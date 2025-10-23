import numpy as np
import copy
from operator import itemgetter
# from collections import defaultdict


#模仿策略函数
def rollout_policy_fn(board):
    '''
    a coarse, fast version of policy_fn used in the rollout phase.
    在蒙特卡洛树搜索（MCTS）中，rollout_policy_fn 是一个用于 rollout 阶段 的简单的策略函数
    它的主要目的是在模拟对局时，以较低的计算成本快速生成动作，从而评估当前局面的胜率。
    '''
    action_probs = np.random.rand(len(board.availables)) # rollout randomly
    return zip(board.availables, action_probs)
#策略模仿价值函数
def policy_value_fn(board):
    '''
    a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state
    '''
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0#将合法动作列表 board.availables 和均匀概率 action_probs 组合成 (动作, 概率) 元组的迭代器，返回state评分为 0

class TreeNode(object):
    '''
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    在MCTS树中的一个节点。每个节点会跟踪它自己的值Q、先验概率P，以及其基于访问次数的先验得分u
    '''

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p # its the prior probability that action's taken to get this node在该节点某个动作被使用的可能性

    def expand(self, action_priors):
        '''
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
        according to the policy function.
        根据策略函数，通过创建新的子节点来扩展树。
        action_priors：一个包含动作及其先验概率的元组列表。
        '''
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
        # expand all children that under this state 扩展当前状态下的所有子节点。

    def select(self, c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        从子节点中选择一个动作，该动作使动作值 Q 加上奖励 u(P) 最大化。
        返回：一个包含 (动作, 下一个节点) 的元组。
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))#从 (动作, 子节点) 元组中提取子节点，并调用其 get_value(c_puct) 方法。
        # self._children is a dict是当前节点的子节点字典
        # act_node[1].get_value will return the action with max Q+u and corresponding state

    def update(self, leaf_value):
        '''
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.
        根据叶子节点的评估值更新节点值。
        leaf_value：从当前玩家的角度评估的子树值。
        '''
        self._n_visits += 1
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # update Q, a running average of values for all visits.
        # there is just: (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)增量平均公式，用于在不存储所有历史数据的情况下，动态计算均值。

    def update_recursive(self, leaf_value):
        '''
        Like a call to update(), but applied recursively for all ancestors.
        类似于 update() 的调用，但递归地应用于所有祖先节点。
        '''
        # If it is not root, this node's parent should be updated first.如果当前节点不是根节点，则应先更新其父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
            # every step for revursive update,
            # we should change the perspective by the way of taking the negative在递归更新的每一步中，我们应该通过取反的方式来切换视角。
        self.update(leaf_value)

    def get_value(self, c_puct):
        '''
        Calculate and return the value for this node.
        It is a combination of leaf evaluations Q,
        and this node's prior adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
        value Q, and prior probability P, on this node's score.
        计算并返回该节点的值。
        它是叶子节点评估值 Q 和基于访问次数调整的先验值 u 的组合。
        c_puct：一个介于 (0, ∞) 之间的数，用于控制值 Q 和先验概率 P 对该节点得分的相对影响。
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''
        check if it's leaf node (i.e. no nodes below this have been expanded).
        '''
        return self._children == {}

    def is_root(self):
        '''
        check if it's root node
        '''
        return self._parent is None

class MCTS(object):
    '''
    A simple implementation of Monte Carlo Tree Search.
    蒙特卡洛树搜索的一个简单实现。
    '''
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400):
        '''
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
            policy_value_fn：一个函数，输入棋盘状态，输出一个包含 (动作, 概率) 元组的列表，
            以及一个介于 [-1, 1] 之间的评分（即从当前玩家的角度评估的终局得分的期望值）。
            c_puct：一个介于 (0, ∞) 之间的数，用于控制探索收敛到最大值策略的速度。值越大，表示越依赖先验概率。
        '''
        self._root = TreeNode(parent=None, prior_p=1.0)
        # root node do not have parent ,and sure with prior probability 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout # times of tree search

    def _playout(self, state):
        '''
        Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        从根节点到叶子节点运行一次模拟，获取叶子节点的值，并将其传播回其父节点。
        state会被就地修改，因此必须提供一个副本。
        '''
        node = self._root
        #选择
        while(1):
            # select action in tree
            if node.is_leaf():
                # break if the node is leaf node
                # print('breaking...................................')
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # print('select action is ...',action)
            # print(action,state.availables)
            state.do_move(action)
            # this state should be the same state with current node这个状态应该与当前节点的状态一致。
        # 扩展
        action_probs, _ = self._policy(state)
        # Check for end of game
        end, winner = state.game_end()
        if not end:
            # expand the node
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal. 反向传播
        node.update_recursive(-leaf_value)
        # print('after update...', node._n_visits, node._Q)

    def _evaluate_rollout(self, state, limit=1000):
        '''
        Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        使用rollout policy玩到游戏结束，
        如果当前玩家获胜则返回+1，如果对手获胜则返回-1，
        如果平局，则为0。
        max(action_probs, key=itemgetter(1))
        max 函数用于从 action_probs 列表中找到最大值。
        key=itemgetter(1) 指定了比较的键，即每个元组的第二个元素（概率）。
        '''
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]#这行代码的作用是从 action_probs 列表中找到概率最大的动作。
            # itemgetter
            # https://www.cnblogs.com/zhoufankui/p/6274172.html
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        # print('winner is ...',winner)
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        '''
        Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        按顺序运行所有playouts，并返回访问量最大的动作。
        state：当前游戏状态
        返回：所选操作
        '''
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
            # use deepcopy and playout on the copy state

        # some statistics just for check
        # visits_count = defaultdict(int)
        # visits_count_dic = defaultdict(int)
        # self.sum = 0
        # Q_U_dic = defaultdict(int)
        # for act,node in self._root._children.items():
        #     visits_count[act] += node._n_visits
        #     visits_count_dic[str(state.move_to_location(act))] += node._n_visits
        #     self.sum += node._n_visits
        #     Q_U_dic[str(state.move_to_location(act))] = node.get_value(5)

        # print(Q_U_dic)
        # print(self.sum,visits_count_dic)

        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        '''
        Step forward in the tree, keeping everything we already know about the subtree.
        保持已探索的子树信息，以减少重复计算。
        '''
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer(object):
    '''
    AI player based on MCTS
    '''
    def __init__(self, c_puct=5, n_playout=400):
        '''
        init a mcts class
        c_puct=5：探索系数，影响 MCTS UCT 公式的探索倾向。
        n_playout=400：模拟次数，表示在每次选择动作时运行 400 次 MCTS 进行评估。
        policy_value_fn：策略评估函数（应该在 MCTS 类中定义）。
        '''
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        '''
        set player index
        记录该 AI 玩家是先手（p=0）还是后手（p=1）
        '''
        self.player = p

    def reset_player(self):
        '''
        reset player
        重新初始化 MCTS 搜索树。
        '''
        self.mcts.update_with_move(-1) # reset the node

    def get_action(self, board,is_selfplay=False,print_probs_value=0):
        '''
        get an action by mcts
        do not discard all the tree and retain the useful part
        通过 MCTS 获取最佳行动
        仅保留有用的部分，而不丢弃整个搜索树
        '''
        sensible_moves = board.availables # 获取可用的合法落子点
        if board.last_move!=-1:
            self.mcts.update_with_move(last_move=board.last_move)
            # reuse the tree重新利用这一步的树
            # retain the tree that can continue to use
            # so update the tree with opponent's move and do mcts from the current node
            # 例如，如果上一步对手落子在 (3, 3)，则搜索树保留这个动作的子树，丢弃其他部分。

        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)# 通过 MCTS 获取最佳落子
            self.mcts.update_with_move(move)# 更新搜索树
            # every time when get a move, update the tree
            #有两次更新，因为人不会更新，这是AI
        else:
            print("WARNING: the board is full")

        return move, None

    def __str__(self):
        return "MCTS {}".format(self.player)








