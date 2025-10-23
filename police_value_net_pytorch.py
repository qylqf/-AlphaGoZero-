import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class PolicyValueNet:
    def __init__(self, board_width, board_height, block,lr = 1e-4, init_model=None, transfer_model=None, cuda=False):
         #board_width 和 board_height：棋盘的宽度和高度。
        # block：残差网络（ResNet）的块数。
        # init_model 和 transfer_model：分别用于加载已有的模型或者迁移学习模型。
        # cuda：是否使用 GPU 进行计算。
        print()
        print('building network ...')
        print()

        self.planes_num = 9 # feature planes
        self.nb_block = block # resnet blocks
        # 是否使用 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.board_width = board_width
        self.board_height = board_height


        # Make a session
        #self.session = tf.InteractiveSession()
        #tf.InteractiveSession() 是 TensorFlow 1.x 中用于创建交互式会话（session）的方式，
        # 适用于需要在 Python 交互环境（如 Jupyter Notebook）中执行 TensorFlow 操作的情况。
        # 1. Input:
        # 是输入的状态张量，表示当前棋盘的状态,可以在运行时动态传入数据
        # self.input_states
            # = tf.placeholder(
            # tf.float32, shape=[None, self.planes_num, board_height, board_width])
        #策略网络输出（即每个位置落子的概率）。价值网络输出（即当前局势的评估值）。reuse=False 和 reuse=True：分别用于训练和测试（共享参数）。
        self.policy_value_net = Network(board_width,board_height)
        self.policy_value_net = self.policy_value_net.to(self.device)  # 移动到gpu
        # self.optimizer = optim.Adam(lr=lr, params=self.policy_value_net.parameters(), weight_decay=1e-4)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)


        if init_model is not None:
            self.restore_model(init_model)
            print('model loaded!')
        elif transfer_model is not None:
            self.restore_model(init_model)
            print('transfer model loaded !')
        else:
            print('can not find saved model, learn from scratch !')
        self.policy_value_train_oppo = Network(board_width, board_height)


    def save_numpy(self, params):
        '''
        Save the model in numpy form
        将 PyTorch 模型的参数保存为 NumPy 数组的形式
        '''
        print('saving model as numpy form ...')
        param = []
        for each in params:
            param.append(each.detach().cpu().numpy())  # Convert tensor to numpy array
        param = np.array(param)  # Convert list to numpy array
        np.save('tmp/model.npy', param)  # Save the numpy array as a .npy file

    def load_numpy(self, params, path='tmp/model.npy'):
        '''
        Load model from numpy
        加载 NumPy 数组形式的模型参数
        '''
        print('loading model from numpy form ...')
        mat = np.load(path)  # Load the numpy file containing model parameters
        for ind, each in enumerate(params):
            each.data = torch.tensor(mat[ind])  # Convert numpy array back to tensor and assign it
        print('load model from numpy!')

    def print_params(self, params):
        # Only for debugging
        return [param.data for param in params]

    def policy_value(self, state_batch, player = 0):
        '''
        input: a batch of states,actin_fc,evaluation_fc
        output: a batch of action probabilities and state values
        输入一批状态（state_batch），通过神经网络计算每个状态的动作概率分布和状态价值。
        输出动作概率分布（act_probs）和状态价值（value）。
        state_batch：一批棋盘状态，通常是经过预处理的张量。
        '''
        model = self.policy_value_net
        model.eval()
        # a = model.named_parameters()
        # b = 0
        # for name, value in a:
        #     if(name == 'residual_blocks.0.conv1.weight'):
        #         b = value
        #
        #
        # b = b.data.detach().cpu().numpy()
        # b = np.transpose(b, (2, 3, 1, 0))
        # state_batch = np.array(state_batch, dtype=np.float32)
        state_batch = torch.tensor(state_batch, dtype=torch.float32)

        state_batch = state_batch.to(self.device)#移动到gpu
        log_act_probs, value = model(state_batch)

        act_probs = np.exp(log_act_probs.detach().cpu().numpy())#shape=(1,width*height)
        return act_probs, value

    def policy_value_fn(self, board):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        '''
        # the accurate policy value fn,
        # i prefer to use one that has some randomness even when test,
        # so that each game can play some different moves, all are ok here
        # #准确的策略值fn，
        # #我更喜欢使用即使在测试时也有一些随机性的，
        # #这样每个游戏都可以玩一些不同的动作，这里都可以
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))
        act_probs, value = self.policy_value(current_state)
        act_probs = zip(legal_positions, act_probs[0][legal_positions])
        return act_probs, value

    def policy_value_fn_random(self, board):
        '''
        input: board,actin_fc,evaluation_fc
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        输入当前棋盘状态（board），通过数据增强（旋转和翻转）生成增强后的状态，
        输出合法动作及其概率，以及当前棋盘状态的价值。
        board：当前棋盘对象，包含棋盘状态和合法动作信息。
        '''
        # like paper said,
        # The leaf node sL is added to a queue for neural network
        # evaluation, (di(p), v) = fθ(di(sL)),
        # where di is a dihedral reflection or rotation
        # selected uniformly at random from i in [1..8]
        #获取当前棋盘的所有合法动作位置。
        # current_state：将棋盘状态转换为适合神经网络输入的张量形式。
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, self.planes_num, self.board_width, self.board_height))

        # print('current state shape',current_state.shape)

        #add dihedral reflection or rotation
        # 数据增强：
        #随机生成旋转角度（rotate_angle）和是否翻转（flip）。
        #对棋盘状态进行旋转和翻转，生成增强后的状态（equi_state）
        rotate_angle = np.random.randint(1, 5)
        flip = np.random.randint(0, 2)
        #按理说无论旋转到哪个方向都是下的棋的相对位置是一样的，所以可以进行数据集的扩充
        equi_state = np.array([np.rot90(s, rotate_angle) for s in current_state[0]])
        if flip:
            equi_state = np.array([np.fliplr(s) for s in equi_state])
        # print(equi_state.shape)

        # put equi_state to network,
        # 将增强后的状态输入神经网络，调用 policy_value 方法计算动作概率分布和状态价值
        act_probs, value = self.policy_value(state_batch=np.array([equi_state]))

        # get dihedral reflection or rotation back
        #将动作空间重塑为self.board_height, self.board_width，并沿着数组的第一个轴（通常是垂直方向）翻转数组中的元素顺序
        #还原动作概率分布：
        #将增强后的动作概率分布还原到原始棋盘状态。
        #通过 np.flipud、np.fliplr 和 np.rot90 等操作，将概率分布恢复到原始方向
        #act_probs[0] 是 act_probs 的第一个（也是唯一一个）样本的动作概率分布，
        # 形状为 (board_width * board_height,)，
        #通常为 1，因为每次只处理一个棋盘状态
        equi_mcts_prob = np.flipud(act_probs[0].reshape(self.board_height, self.board_width))
        if flip:
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
        equi_mcts_prob = np.rot90(equi_mcts_prob, 4 - rotate_angle)
        act_probs = np.flipud(equi_mcts_prob).flatten()
        #返回结果：
        # #将合法动作及其概率打包成 (action, probability) 的元组列表。
        # #返回动作概率列表和状态价值。
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, optimizer, state_batch, mcts_probs, winner_batch):
        '''
        perform a training step
        winner_batch：每个棋盘状态的胜负标签（1 表示当前玩家获胜，-1 表示对手获胜，0 表示平局）

        '''
        # 将数据转换为 PyTorch 张量
        model = self.policy_value_net
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        mcts_probs = torch.tensor(np.array(mcts_probs), dtype=torch.float32)
        winner_batch = torch.tensor(winner_batch, dtype=torch.float32).view(-1, 1)
        # 移动到gpu
        state_batch = state_batch.to(self.device)
        winner_batch = winner_batch.to(self.device)
        mcts_probs = mcts_probs.to(self.device)
        #开始训练
        model.train()
        #参数更新
        optimizer.zero_grad()
        action_fc_train, evaluation_fc2_train = model(state_batch)
        loss = self.compute_loss(winner_batch, mcts_probs,
                                 action_fc_train, evaluation_fc2_train)
        loss.backward()

        #梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        optimizer.step()
        entropy = self.entropy(action_fc_train)
        return loss, entropy

    def save_model(self, model_path):
        '''
            保存模型参数到指定路径
            '''
        # 保存模型的 state_dict
        model = self.policy_value_net
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
    def restore_model(self, model_path, layer_names=None):
        '''
        从给定路径加载模型权重
        可以选择性加载部分层，例如卷积层或残差块
        '''
        model = self.policy_value_net
        model_dict = model.state_dict()
        checkpoint = torch.load(model_path)

        # # 如果需要只恢复特定层
        # if layer_names:
        #     # 只加载指定层的权重
        #     for name, param in checkpoint.items():
        #         if name in layer_names:
        #             model_dict[name] = param
        # else:
        #     # 加载所有权重
        #     model_dict.update(checkpoint)

        model.load_state_dict(torch.load(model_path))
        model.eval()  # 切换到评估模式
        print(f"Model restored from {model_path}")
        return model

    def entropy (self, action_fc_test):
        return -torch.mean(torch.sum(torch.exp(action_fc_test) * action_fc_test, dim=1))

    def compute_loss(self, labels, mcts_probs, action_fc_train, evaluation_fc2_train, l2_penalty_beta = 1e-4):
        model = self.policy_value_net
        # 1. Value Loss (均方误差)
        value_loss = nn.MSELoss()(evaluation_fc2_train, labels)
        # 2. Policy Loss (交叉熵)
        policy_loss = -torch.mean(torch.sum(mcts_probs * action_fc_train, dim=1))
        # 3. L2正则化
        # l2_penalty_beta = 1e-4
        # l2_loss = 0.0
        # for name, param in model.named_parameters():
        #     if 'bias' not in name.lower():
        #         l2_loss += torch.norm(param, 2) ** 2
        # l2_penalty = l2_penalty_beta * l2_loss
        # # 4. 总损失
        total_loss = value_loss + policy_loss
        return total_loss

class Network(nn.Module):
    """
    定义策略-价值网络
    """
    def __init__(self, board_width, board_height, num_residual_blocks=19):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        # self.zero_pad = nn.ZeroPad2d(padding=2)
        # 输入层
        self.conv1 = nn.Conv2d(9, 64, kernel_size=1, stride=1, padding=0)
        # self.bn1 = nn.BatchNorm2d(64)

        # 残差块
        self.residual_blocks = nn.Sequential(
            *[Residual(64, 64) for _ in range(num_residual_blocks)]
        )

        # 动作网络
        self.action_conv = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        self.action_bn = nn.BatchNorm2d(2)
        self.flatten = nn.Flatten()
        self.action_fc = nn.Linear(2 * board_width * board_height, board_width * board_height)

        # 评估网络
        self.evaluation_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        self.evaluation_bn = nn.BatchNorm2d(1)
        self.evaluation_fc1 = nn.Linear(board_width * board_height, 256)
        self.evaluation_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x = self.zero_pad(x)
        # 公共网络层

        x = self.residual_blocks(F.relu(self.conv1(x)))

        # 动作网络
        action = F.relu(self.action_bn(self.action_conv(x)))
        action = self.flatten(action)  # 展平
        action_prob = F.log_softmax(self.action_fc(action), dim=1)

        # 评估网络
        evaluation = F.relu(self.evaluation_bn(self.evaluation_conv(x)))
        evaluation = self.flatten(evaluation)  # 展平
        evaluation = F.relu(self.evaluation_fc1(evaluation))
        evaluation_value = torch.tanh(self.evaluation_fc2(evaluation))

        return action_prob, evaluation_value

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):  # num_channels为输出channel数
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1,
                               stride=strides, bias=True)  # 可以使用传入进来的strides
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1,
                               stride=strides, bias=True)  # 使用nn.Conv2d默认的strides=1
        # if use_1x1conv:
        #     self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        # else:
        #     self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()  # inplace原地操作，不创建新变量，对原变量操作，节约内存

    def forward(self, X):

        Y = F.relu(self.bn1(self.conv1(X)))

        Y = self.bn2(self.conv2(Y))
        # if self.conv3:
        #     X = self.conv3(X)
        Y = Y +  X

        return F.relu(Y)



