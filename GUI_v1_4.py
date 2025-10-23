"""
introduction:
    This is a UI for Gomoku game which does not contain the rule code.

    Set self.UnitSize in __init__() to a different value can change the basic size of all elements

    There are no limits for the value of board_size. So a board of any size can be created(if the system supports).
    Add limits in self.reset() if necessary.
    这是一个用于五子棋游戏的用户界面，但不包含规则代码。

在 __init__() 中设置 self.UnitSize 为不同的值，可以改变所有元素的基本大小。

board_size 的值没有限制。所以可以创建任意大小的棋盘（前提是系统支持）。如有必要，可以在 self.reset() 中添加限制。
"""
import pygame
from pygame.locals import *


class GUI:

    def __init__(self, board_size=11):
        '''
        初始化游戏界面，设置基本参数（棋盘尺寸、单位尺寸、颜色等），创建Pygame窗口，
        并调用reset()、restart_game()和reset_score()完成初始布局
        '''
        pygame.init()

        self.score = [0, 0] #记录两名玩家的得分
        self.BoardSize = board_size #棋盘大小
        self.UnitSize = 40      # the basic size of all elements, try a different value!所有 UI 元素的基础尺寸单位（像素）
        self.TestSize = int(self.UnitSize * 0.625)#文字的基础大小，基于 UnitSize 计算（默认 25 像素）
        self.state = {}         # a dictionary for pieces on board. filled with move-player pairs, such as 34:1字典，记录棋盘上的落子位置和玩家
        self.areas = {}         # a dictionary for button areas.
        # filled with name-Rect pairs存储界面中按钮和棋盘的矩形区域（PyGame 的 Rect 对象）用于检测鼠标点击位置（如判断点击的是按钮还是棋盘）
        #名字：按钮范围
        self.ScreenSize = None  # save the screen size for some calculation窗口尺寸（根据 UnitSize 和 BoardSize 计算）
        self.screen = None#PyGame 窗口对象，用于绘制所有图形
        self.last_action_player = None#记录上一步操作的玩家和位置（格式 (move, player)）
        self.round_counter = 0#当前回合数，用于统计游戏轮次
        self.messages = ''#界面底部显示的临时消息（如回合提示）
        self._background_color = (197, 227, 205)#界面背景颜色（RGB 值，浅绿色）
        self._board_color = (254, 185, 120)#棋盘颜色（RGB 值，橙黄色）

        self.reset(board_size)

        # restart_game() must be called before reset_score() because restart_game() will add value to self.round_counter
        self.restart_game(False)
        self.reset_score()

    def reset(self, bs):
        """
        reset screen
        :param bs: board size
        重置屏幕和棋盘尺寸，根据新的棋盘大小调整窗口尺寸，
        并初始化按钮区域（如“重新开始”、“切换玩家”按钮）和棋盘区域。
        bs: 新的棋盘尺寸（支持任意大小，但可通过注释添加限制）
        """

        # # you can add limits for board size
        # bs = int(bs)
        # if bs < 5:
        #     raise ValueError('board size too small')

        self.BoardSize = bs
        self.ScreenSize = (self.BoardSize * self.UnitSize + 2 * self.UnitSize,
                           self.BoardSize * self.UnitSize + 3 * self.UnitSize)
        self.screen = pygame.display.set_mode(self.ScreenSize, 0, 32)
        pygame.display.set_caption('AlphaZero_Gomoku')

        # button areas
        self.areas['SwitchPlayer'] = Rect(self.ScreenSize[0]/2-self.UnitSize*1.5, self.ScreenSize[1] - self.UnitSize, self.UnitSize*3, self.UnitSize)
        self.areas['RestartGame'] = Rect(self.ScreenSize[0] - self.UnitSize*3, self.ScreenSize[1] - self.UnitSize, self.UnitSize*3, self.UnitSize)
        self.areas['ResetScore'] = Rect(0, self.ScreenSize[1] - self.UnitSize, self.UnitSize*2.5, self.UnitSize)

        board_lenth = self.UnitSize * self.BoardSize
        self.areas['board'] = Rect(self.UnitSize, self.UnitSize, board_lenth, board_lenth)

    def restart_game(self, button_down=True):
        """
        restart for a new round
        :param button_down: whether the RestartGame button is pressed, used to highlight button.
        开始新一局游戏。重置棋盘状态（清空棋子），更新回合计数器，并重绘静态元素（棋盘、按钮）
        button_down: 是否按下“重新开始”按钮（用于按钮高亮效果）
        """
        self.round_counter += 1
        self._draw_static()
        if button_down:
            self._draw_button('RestartGame', 1)
        self.state = {}
        self.last_action_player = None
        pygame.display.update()

    def reset_score(self):
        """
        reset score and round
        重置玩家分数和回合计数器，并通过 show_messages() 更新界面显示。
        """
        self.score = [0, 0]
        self.round_counter = 1
        self.show_messages()

    def add_score(self, winner):
        """
        add score for winner
        :param winner: the name of the winner
        根据胜者更新分数，并刷新界面显示。
        winner: 胜者编号（1 或 2）
        """
        if winner == 1:
            self.score[0] += 1
        elif winner == 2:
            self.score[1] += 1
        else:
            raise ValueError('player number error')
        self.show_messages()

    def render_step(self, action, player):
        """
        render a step of the game
        :param action: 1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player
        渲染玩家的一步操作。绘制棋子并在最后一步的棋子上添加十字标记
        action: 落子位置（坐标或一维编号）。
        player: 当前玩家编号（1 或 2）。
        """
        try:
            action = int(action)
        except Exception:
            pass
        if type(action) != int:
            move = self.loc_2_move(action)
        else:
            move = action

        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

        if self.last_action_player:     # draw a cross on the last piece to mark the last move
            self._draw_pieces(self.last_action_player[0], self.last_action_player[1], False)

        self._draw_pieces(action, player, True)
        self.state[move] = player
        self.last_action_player = move, player

    def move_2_loc(self, move):
        """
        transfer a move value to a location value
        :param move: an int type move value such as 34
        :return: an 1*2 dimension location value such as (2, 3)
        一维编号与二维坐标互相转换
        """
        return move % self.BoardSize, move // self.BoardSize

    def loc_2_move(self, loc):
        """
        transfer a move value to a location value
        :param loc: an 1*2 dimension location value such as (2, 3)
        :return: an int type move value such as 34
        一维编号与二维坐标互相转换
        """
        return loc[0] + loc[1] * self.BoardSize

    def get_input(self):
        """
        get inputs from clicks
        :return: variable-length array.[0] is the name. Additional information behind (maybe not exist).
        监听鼠标事件，返回用户操作类型（如点击按钮、落子、退出）
        如 ('move', 34) 表示落子，('RestartGame',) 表示点击重启按钮
        """
        while True:
            event = pygame.event.wait()
            if event.type == QUIT:
                return 'quit',

            if event.type == MOUSEBUTTONDOWN:   # check mouse click event
                if event.button == 1:
                    mouse_pos = event.pos

                    for name, rec in self.areas.items():
                        if self._in_area(mouse_pos, rec):
                            if name != 'board':
                                self._draw_button(name, 2, True)
                                pygame.time.delay(100)
                                self._draw_button(name, 1, True)
                                return name,
                            else:
                                x = (mouse_pos[0] - self.UnitSize)//self.UnitSize
                                y = self.BoardSize - (mouse_pos[1] - self.UnitSize)//self.UnitSize - 1
                                move = self.loc_2_move((x, y))
                                if move not in self.state:
                                    return 'move', move

            if event.type == MOUSEMOTION:       # check mouse move event to highlight buttons
                mouse_pos = event.pos
                for name, rec in self.areas.items():
                    if name != 'board':
                        if self._in_area(mouse_pos, rec):
                            self._draw_button(name, 1, True)
                        else:
                            self._draw_button(name, update=True)

    def deal_with_input(self, inp, player):
        """
        This is just a example to deal with inputs
        :param inp: inputs from get_input()
        :param player: the name of the player
        根据输入类型处理逻辑（示例方法，需用户扩展）。例如重启游戏、处理落子
        inp: get_input() 的返回值。
        player: 当前玩家编号。
        """
        if inp[0] == 'RestartGame':
            self.restart_game()
        elif inp[0] == 'ResetScore':
            self.reset_score()
        elif inp[0] == 'quit':
            exit()
        elif inp[0] == 'move':
            self.render_step(inp[1], player)
        elif inp[0] == 'SwitchPlayer':
            # restart_game() must be called before reset_score(). The reason is mentioned above.
            UI.restart_game(False)
            UI.reset_score()
            # code for switch is needed

    def show_messages(self, messages=None):
        """
        show extra messages on screen
        :param messages:
        :return:
        在界面底部显示自定义消息（如回合提示），并更新分数和回合数
        messages: 自定义文本（可选，默认显示当前消息）
        """
        if messages:
            self.messages = messages
        pygame.draw.rect(self.screen, self._background_color, (0, self.ScreenSize[1]-self.UnitSize*2, self.ScreenSize[0], self.UnitSize))
        self._draw_round(False)
        self._draw_text(self.messages, (self.ScreenSize[0]/2, self.ScreenSize[1]-self.UnitSize*1.5), text_height=self.TestSize)
        self._draw_score()

    def _draw_score(self, update=True):
        #绘制分数到界面指定位置。
        score = 'Score: ' + str(self.score[0]) + ' : ' + str(self.score[1])
        self._draw_text(score, (self.ScreenSize[0] * 0.11, self.ScreenSize[1] - self.UnitSize*1.5),
                        backgroud_color=self._background_color, text_height=self.TestSize)
        if update:
            pygame.display.update()

    def _draw_round(self, update=True):
        #绘制回合数到界面指定位置。
        self._draw_text('Round: ' + str(self.round_counter), (self.ScreenSize[0]*0.88, self.ScreenSize[1] - self.UnitSize*1.5),
                        backgroud_color=self._background_color, text_height=self.TestSize)
        if update:
            pygame.display.update()

    def _draw_pieces(self, loc, player, last_step=False):
        """
        draw pieces
        :param loc:  1*2 dimension location value such as (2, 3) or an int type move value such as 34
        :param player: the name of the player
        :param last_step: whether it is the last step
        根据玩家编号绘制黑色或白色棋子，并在最后一步添加十字标记
        """
        try:
            loc = int(loc)
        except Exception:
            pass

        if type(loc) is int:
            x, y = self.move_2_loc(loc)
        else:
            x, y = loc
        pos = int(self.UnitSize * 1.5 + x * self.UnitSize), int(self.UnitSize * 1.5 + (self.BoardSize - y - 1) * self.UnitSize)
        if player == 1:
            c = (0, 0, 0)
        elif player == 2:
            c = (255, 255, 255)
        else:
            raise ValueError('num input ValueError')
        pygame.draw.circle(self.screen, c, pos, int(self.UnitSize * 0.45))
        if last_step:
            if player == 1:
                c = (255, 255, 255)
            elif player == 2:
                c = (0, 0, 0)

            start_p1 = pos[0] - self.UnitSize * 0.3, pos[1]
            end_p1 = pos[0] + self.UnitSize * 0.3, pos[1]
            pygame.draw.line(self.screen, c, start_p1, end_p1)

            start_p2 = pos[0], pos[1] - self.UnitSize * 0.3
            end_p2 = pos[0], pos[1] + self.UnitSize * 0.3
            pygame.draw.line(self.screen, c, start_p2, end_p2)

    def _draw_static(self):
        """
        Draw static elements that will not change in a round.
        绘制静态元素（棋盘背景、网格线、坐标轴、按钮）
        """
        # draw background
        self.screen.fill(self._background_color)
        # draw board
        board_lenth = self.UnitSize * self.BoardSize
        pygame.draw.rect(self.screen, self._board_color, self.areas['board'])
        for i in range(self.BoardSize):
            # draw grid lines
            start = self.UnitSize * (i + 0.5)
            pygame.draw.line(self.screen, (0, 0, 0), (start + self.UnitSize, self.UnitSize*1.5),
                             (start + self.UnitSize, board_lenth + self.UnitSize*0.5))
            pygame.draw.line(self.screen, (0, 0, 0), (self.UnitSize*1.5, start + self.UnitSize),
                             (board_lenth + self.UnitSize*0.5, start + self.UnitSize))
            pygame.draw.rect(self.screen, (0, 0, 0), (self.UnitSize, self.UnitSize, board_lenth, board_lenth), 1)
            # coordinate values
            self._draw_text(self.BoardSize - i - 1, (self.UnitSize / 2, start + self.UnitSize), text_height=self.TestSize)  # 竖的
            self._draw_text(i, (start + self.UnitSize, self.UnitSize / 2), text_height=self.TestSize)  # 横的

        # draw buttons
        for name in self.areas.keys():
            if name != 'board':
                self._draw_button(name)

        self.show_messages()

    def _draw_text(self, text, position, text_height=25, font_color=(0, 0, 0), backgroud_color=None, pos='center',
                   angle=0):
        """
        draw text
        :param text: a string type text
        :param position: the location point
        :param text_height: text height
        :param font_color: font color
        :param backgroud_color: background color
        :param pos: the location point is where in the text rectangle.
        'center','top','bottom','left','right'and their combination such as 'topleft' can be selected
        :param angle: the rotation angle of the text
        在指定位置绘制文本，支持旋转和背景色。
        """
        posx, posy = position
        font_obj = pygame.font.Font(None, int(text_height))
        text_surface_obj = font_obj.render(str(text), True, font_color, backgroud_color)
        text_surface_obj = pygame.transform.rotate(text_surface_obj, angle)
        text_rect_obj = text_surface_obj.get_rect()
        exec('text_rect_obj.' + pos + ' = (posx, posy)')
        self.screen.blit(text_surface_obj, text_rect_obj)

    def _draw_button(self, name, high_light=0, update=False):
        #绘制按钮，支持高亮效果（未按下/悬停/按下）
        rec = self.areas[name]
        if not high_light:
            color = (225, 225, 225)
        elif high_light == 1:
            color = (245, 245, 245)
        elif high_light == 2:
            color = (255, 255, 255)
        else:
            raise ValueError('high_light value error')
        pygame.draw.rect(self.screen, color, rec)
        pygame.draw.rect(self.screen, (0, 0, 0), rec, 1)
        self._draw_text(name, rec.center, text_height=self.TestSize)
        if update:
            pygame.display.update()

    @staticmethod
    def _in_area(loc, area):
        """
        check whether the location is in area
        :param loc: a 1*2 dimension location value such as (123, 45)
        :param area: a Rect type value in pygame
        静态方法，判断鼠标位置 loc 是否在矩形区域 area 内
        """
        return True if area[0] < loc[0] < area[0] + area[2] and area[1] < loc[1] < area[1] + area[3] else False


if __name__ == '__main__':
    #创建一个 GUI 实例并模拟游戏循环，处理用户输入和界面更新。
    # test
    UI = GUI()
    action = 22
    player = 1
    i = 1
    UI.add_score(1)
    while True:
        if i == 1:
            UI.show_messages('first player\'s turn')
        else:
            UI.show_messages('second player\'s turn')
        inp = UI.get_input()
        print(inp)
        UI.deal_with_input(inp, i)
        if inp[0] == 'move':
            i %= 2
            i += 1
        elif inp[0] == 'RestartGame':
            i = 1
        elif inp[0] == 'SwitchPlayer':
            i = 1
