from __future__ import print_function
# import random
import numpy as np
import os
import time
from collections import defaultdict, deque
from game_board import Board,Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from police_value_net_pytorch import PolicyValueNet
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
# import random
# import torch
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

class TrainPipeline():
    def __init__(self, init_model=None, transfer_model=None):

        # 添加历史记录变量
        self.loss_history = []
        self.kl_history = []
        self.entropy_history = []
        self.explained_var_old_history = []
        self.explained_var_new_history = []
        self.win_ratio_history = []
        self.eval_steps = []  # 记录评估点的训练步数

        self.resnet_block = 19  # num of block structures in resnet
        # params of the board and the game
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # training params
        self.learn_rate = 1e-1
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000 # memory size经验回放缓冲区的大小。
        self.batch_size = 512  # mini-batch size for training训练时的批量大小。
        self.data_buffer = deque(maxlen=self.buffer_size)#用于存储训练数据的队列。
        self.play_batch_size = 1 # play n games for each network training每次训练前自我对弈的游戏数量。
        self.check_freq = 100#模型评估频率。
        self.game_batch_num = 400 # total game to train总训练游戏数量。
        self.best_win_ratio = 0.0#最佳胜率（用于模型保存）。
        # num of simulations used for the pure mcts, which is used as纯 MCTS 的模拟次数（用于评估模型）。
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 3000#评价的纯 MCTS 的模拟次数
        #初始化策略-价值网络,如果提供了初始模型或预训练模型，则加载模型；否则从头开始训练。
        if (init_model is not None) and os.path.exists(init_model+'.index'):
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height, block=self.resnet_block, init_model=init_model, cuda=True)
        elif (transfer_model is not None) and os.path.exists(transfer_model+'.index'):
            # start training from a pre-trained policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height,block=self.resnet_block, transfer_model=transfer_model, cuda=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,self.board_height, block=self.resnet_block, init_model="tmp\\current_policy.model", cuda=True)
#初始化 MCTS 玩家

        self.mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                       policy_value_net=self.policy_value_net.policy_value_net,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout,
                                       is_selfplay=True)
        self.optimizer = optim.AdamW(lr=1e-2, params=self.policy_value_net.policy_value_net.parameters(), weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)

    def get_equi_data(self, play_data):
        '''
        数据增强
        通过旋转和翻转扩充数据集，增加数据的多样性。
        使用 np.rot90 和 np.fliplr 实现数据增强。
        augment the data set by rotation and flipping
        通过旋转和翻转来扩充数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        原始训练数据，格式为 [(state, mcts_prob, winner_z), ...]
        '''
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                #rotate counterclockwise 90*i
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                #np.flipud like A[::-1,...]
                #https://docs.scipy.org/doc/numpy-1.6.0/reference/generated/numpy.flipud.html
                # change the reshaped numpy
                # 0,1,2,
                # 3,4,5,
                # 6,7,8,
                # as
                # 6 7 8
                # 3 4 5
                # 0 1 2
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                #这个np.fliplr like m[:, ::-1]
                #https://docs.scipy.org/doc/numpy/reference/generated/numpy.fliplr.html
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        '''
        collect self-play data for training
        自我对弈
        使用当前策略生成自我对弈数据。
        数据格式为 (state, mcts_prob, winners_z)，其中：
        state 是棋盘状态。
        mcts_prob 是 MCTS 生成的动作概率。
        z：对弈结果（从当前玩家的视角看，1 表示胜利，-1 表示失败，0 表示平局）
        '''
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        '''
        模型训练
        从数据缓冲区中随机采样小批量数据。
        使用策略-价值网络计算动作概率和状态价值。
        调用 train_step 方法更新模型参数。
        计算 KL 散度、损失、熵和解释方差，监控训练过程。
        update the policy-value net
        self.data_buffer：经验回放缓冲区，存储了自我对弈生成的数据，格式为 [(state, mcts_prob, winner_z), ...]。
        loss：当前训练步骤的损失值。
        entropy：动作概率分布的熵。
        '''
        # play_data: [(state, mcts_prob, winner_z), ..., ...]
        # train an epoch
        epoch_loss = []
        epoch_kl = []
        epoch_entropy = []
        epoch_explained_old = []
        epoch_explained_new = []

        tmp_buffer = np.array(self.data_buffer, dtype=object)#将经验回放缓冲区转换为 NumPy 数组。
        np.random.shuffle(tmp_buffer)#随机打乱数据，确保训练的随机性。
        steps = len(tmp_buffer)//self.batch_size#计算训练步数，即总数据量除以批量大小。
        print('tmp buffer: {}, steps: {}'.format(len(tmp_buffer),steps))


        for i in range(steps):
            mini_batch = tmp_buffer[i*self.batch_size:(i+1)*self.batch_size]#从打乱后的数据中提取一个小批量数据。
            #提取数据
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            mcts_probs_batch = np.array(mcts_probs_batch, dtype=np.float32)
            winner_batch = np.array(winner_batch, dtype=np.float32)
            state_batch = np.array(state_batch, dtype=np.float32)
            #使用当前策略-价值网络计算旧的动作概率和状态值。
            old_probs, old_v = self.policy_value_net.policy_value(state_batch=state_batch)

            #调用 train_step 方法，更新模型参数。
            loss, entropy = self.policy_value_net.train_step(optimizer=self.optimizer,
                                                             state_batch=state_batch,
                                                             mcts_probs=mcts_probs_batch,
                                                             winner_batch=winner_batch)

            # 更新学习率
            self.scheduler.step()
            #使用更新后的策略-价值网络计算新的动作概率和状态值。
            new_probs, new_v = self.policy_value_net.policy_value(state_batch=state_batch)
            #衡量新旧动作概率分布之间的差异。
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            #衡量模型预测值与真实值之间的相关性。

            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.detach().cpu().numpy().flatten()) /
                                 (np.var(np.array(winner_batch)) + 1e-8))
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.detach().cpu().numpy().flatten()) /
                                 (np.var(np.array(winner_batch)) + 1e-8))

            # del loss, entropy, old_probs, old_v  # 删除不再需要的变量
            # torch.cuda.empty_cache()  # 释放未使用的显存

            epoch_loss.append(loss.item())
            epoch_kl.append(kl.item())
            epoch_entropy.append(entropy.item())
            epoch_explained_old.append(explained_var_old.item())
            epoch_explained_new.append(explained_var_new.item())

            if steps<10 or (i%(steps//10)==0):
                # print some information, not too much
                # 收集每个batch的指标
                print('batch: {},length: {}'
                      'kl:{:.5f},'
                      'loss:{},'
                      'entropy:{},'
                      'explained_var_old:{:.3f},'
                      'explained_var_new:{:.3f}'.format(i,
                                                        len(mini_batch),
                                                        kl.item(),
                                                        loss.item(),
                                                        entropy.item(),
                                                        explained_var_old.item(),
                                                        explained_var_new.item()))

                # 记录平均指标
            if epoch_loss:
                self.loss_history.append(np.mean(epoch_loss))
                self.kl_history.append(np.mean(epoch_kl))
                self.entropy_history.append(np.mean(epoch_entropy))
                self.explained_var_old_history.append(np.mean(epoch_explained_old))
                self.explained_var_new_history.append(np.mean(epoch_explained_new))

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        '''
        性能评估
        通过与纯 MCTS 玩家对战评估模型性能。
        计算胜率，保存最佳模型。
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        通过与纯MCTS玩家对战来评估训练策略
        注：这仅用于监控训练进度
        '''
        current_mcts_player = MCTSPlayer(policy_value_function=self.policy_value_net.policy_value_fn_random,
                                       policy_value_net=self.policy_value_net.policy_value_net,
                                       c_puct=5,
                                       n_playout=self.n_playout,
                                       is_selfplay=False)

        test_player = MCTS_Pure(c_puct=5,
                                n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(player1=current_mcts_player,
                                          player2=test_player,
                                          start_player=i % 2,
                                          is_shown=0,
                                          print_prob=False)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def plot_training_curves(self):
        plt.figure(figsize=(15, 10))

        # 训练指标
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history, label='Loss')
        # plt.plot(self.kl_history, label='KL Divergence')
        # plt.plot(self.entropy_history, label='Entropy')
        plt.xlabel('Training Steps')
        plt.title('Training Metrics')
        plt.legend()

        # 解释方差
        plt.subplot(2, 2, 2)
        plt.plot(self.explained_var_old_history, label='Explained Var Old')
        plt.plot(self.explained_var_new_history, label='Explained Var New')
        plt.xlabel('Training Steps')
        plt.title('Explained Variance')
        plt.legend()

        # 胜率
        plt.subplot(2, 2, 3)
        if self.eval_steps:
            plt.plot(self.eval_steps, self.win_ratio_history, 'o-')
            plt.xlabel('Training Steps')
            plt.ylabel('Win Ratio')
            plt.title('Evaluation Win Ratio')
            plt.ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

    def run(self):
        '''
        训练循环
        重复执行数据收集、训练和评估
        定期保存模型和评估性能。
        run the training pipeline
        '''
        # make dirs first创建 tmp 和 model 目录，用于存储临时模型和最佳模型
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        if not os.path.exists('model'):
            os.makedirs('model')

        # record time for each part记录训练开始时间，以及数据收集、训练和评估的时间。
        start_time = time.time()
        collect_data_time = 0
        train_data_time = 0
        evaluate_time = 0

        try:
            for i in range(self.game_batch_num):
                # collect self-play data收集自我对弈数据，累计数据收集时间。
                collect_data_start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                collect_data_time += time.time()-collect_data_start_time
                print("batch i:{}, episode_len:{}, time:{}".format(
                        i+1, self.episode_len, collect_data_time / 60))#打印当前训练批次和每局游戏的长度。

                #训练模型
                if len(self.data_buffer) > self.batch_size*5:
                    # train collected data
                    train_data_start_time = time.time()
                    loss, entropy = self.policy_update()

                    train_data_time += time.time()-train_data_start_time

                    # print some training information
                    print('now time : {}'.format((time.time() - start_time) / 3600))
                    print('collect_data_time : {}, train_data_time : {},evaluate_time : {}'.format(
                        collect_data_time / 3600, train_data_time / 3600,evaluate_time/3600))

                #定期评估模型
                if (i+1) % self.check_freq == 0 :#每隔 self.check_freq 次训练，保存当前模型。

                    # save current model for evaluating
                    self.policy_value_net.save_model(model_path='tmp/current_policy.model')
                    if (i+1) % (self.check_freq*2) == 0:#每隔 self.check_freq*2次训练，评估性能
                        # print("current self-play batch: {}".format(i + 1))
                        evaluate_start_time = time.time()

                        # evaluate current model
                        win_ratio = self.policy_evaluate(n_games=10)
                        self.win_ratio_history.append(win_ratio)
                        self.eval_steps.append(len(self.loss_history))  # 记录当前训练步数
                        evaluate_time += time.time()-evaluate_start_time#累计评估时间。
                        print("current self-play batch: {}, win_ratio:{}".format(i + 1, win_ratio))
                        if win_ratio >= self.best_win_ratio:
                            # save best model
                            print("New best policy!!!!!!!!")
                            self.best_win_ratio = win_ratio
                            self.policy_value_net.save_model(model_path='model/best_policy.model')

                            if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
                                # increase playout num and  reset the win ratio
                                # 如果胜率达到 1.0，则增加纯 MCTS 的模拟次数；如果模拟次数达到 5000，则重置为 1000。
                                # 当纯 MCTS 的模拟次数达到 5000 时，其决策能力已经非常强，模型的胜率可能难以进一步提升。
                                # 重置为 1000 可以降低评估难度，确保模型在较低的评估难度下继续提升性能。
                                self.pure_mcts_playout_num += 500
                                self.best_win_ratio = 0.0
                            if self.pure_mcts_playout_num ==5000:
                                # reset mcts pure playout num
                                self.pure_mcts_playout_num = 1000
                                self.best_win_ratio = 0.0
                        # 训练结束后绘图
            self.plot_training_curves()
        except KeyboardInterrupt:
            self.plot_training_curves()
            print('\n\rquit')

if __name__ == '__main__':
    training_pipeline = TrainPipeline(init_model='model/best_policy.model',transfer_model=None)
    # training_pipeline = TrainPipeline(init_model=None, transfer_model='transfer_model/best_policy.model')
    # training_pipeline = TrainPipeline()
    training_pipeline.run()