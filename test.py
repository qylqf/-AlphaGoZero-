import matplotlib.pyplot as plt

class TrainPipeline():

    def __init__(self, init_model=None, transfer_model=None):
        self.loss_history = []
        self.explained_var_old_history = []
        self.explained_var_new_history = []
        self.eval_steps = [3500,7000]
        self.win_ratio_history = [0.6,0.9]

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

t = TrainPipeline()
t.plot_training_curves()