import numpy as np
import gridworld_render as render_helper
import matplotlib.pyplot as plt
import matplotlib


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # 행동 공간(가능한 행동들)
        self.action_meaning = {  # 행동의 의미
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array(  # 보상 맵(각 좌표의 보상 값)
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3)    # 목표 상태(좌표)
        self.wall_state = (1, 1)    # 벽 상태(좌표)
        self.start_state = (2, 0)   # 시작 상태(좌표)
        self.agent_state = self.start_state   # 에이전트 초기 상태(좌표)

    @property
    def height(self):
        return len(self.reward_map)

    @property
    def width(self):
        return len(self.reward_map[0])

    @property
    def shape(self):
        return self.reward_map.shape

    def actions(self):
        return self.action_space

    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state, action):
        # 이동 위치 계산
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        # 이동한 위치가 그리드 월드의 테두리 밖이나 벽인가?
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state  # 다음 상태 반환

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def step(self, action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)



class FiveGridWorld(GridWorld):
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        
        self.reward_map = np.array([
            [0, 0, 0, -1, 1],
            [0, 0, 0, 0, 0],
            [0, None, None, 0, 0],
            [0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0]
        ])
        
        self.goal_state = (0, 4)
        self.wall_states = [(2, 1), (2, 2)]
        self.start_state = (4, 0)

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state in self.wall_states:
            next_state = state

        return next_state
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                state = (h, w)
                if state not in self.wall_states:
                    yield state
    
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, self.wall_states)
        
        renderer.set_figure()
        
        ys, xs = renderer.ys, renderer.xs
        ax = renderer.ax
        
        if v is not None:
            color_list = ['red', 'white', 'green']
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)
            
            v_dict = v
            v_array = np.zeros(self.reward_map.shape)
            for state, value in v_dict.items():
                v_array[state] = value
            
            vmax, vmin = v_array.max(), v_array.min()
            vmax = max(vmax, abs(vmin))
            vmin = -1 * vmax
            vmax = 1 if vmax < 1 else vmax
            vmin = -1 if vmin > -1 else vmin
            
            ax.pcolormesh(np.flipud(v_array), cmap=cmap, vmin=vmin, vmax=vmax)
        
        for y in range(ys):
            for x in range(xs):
                state = (y, x)
                r = self.reward_map[y, x]
                
                if r != 0 and r is not None:
                    txt = 'R ' + str(r)
                    if state == self.goal_state:
                        txt = txt + ' (GOAL)'
                    ax.text(x+.1, ys-y-0.9, txt)
                
                if (v is not None) and state not in self.wall_states:
                    if print_value:
                        offsets = [(0.4, -0.15), (-0.15, -0.3)]
                        key = 0
                        if v_array.shape[0] > 7: key = 1
                        offset = offsets[key]
                        ax.text(x+offset[0], ys-y+offset[1], "{:12.2f}".format(v_array[y, x]))
                
                if policy is not None and state not in self.wall_states:
                    actions = policy[state]
                    max_actions = [kv[0] for kv in actions.items() if kv[1] == max(actions.values())]
                    
                    arrows = ["↑", "↓", "←", "→"]
                    offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
                    for action in max_actions:
                        arrow = arrows[action]
                        offset = offsets[action]
                        if state == self.goal_state:
                            continue
                        ax.text(x+0.45+offset[0], ys-y-0.5+offset[1], arrow)
                
                if state in self.wall_states:
                    ax.add_patch(plt.Rectangle((x,ys-y-1), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
        
        plt.show()

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state, self.wall_states[0])  # 임시로 첫 번째 wall state 전달
        
        renderer.set_figure()
        
        ys, xs = renderer.ys, renderer.xs
        ax = renderer.ax
        action_space = [0, 1, 2, 3]

        if q is not None:
            qmax, qmin = max(q.values()), min(q.values())
            qmax = max(qmax, abs(qmin))
            qmin = -1 * qmax
            qmax = 1 if qmax < 1 else qmax
            qmin = -1 if qmin > -1 else qmin

            color_list = ['red', 'white', 'green']
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'colormap_name', color_list)

            for y in range(ys):
                for x in range(xs):
                    state = (y, x)
                    r = self.reward_map[y, x]
                    
                    if r != 0 and r is not None:
                        txt = 'R ' + str(r)
                        if state == self.goal_state:
                            txt = txt + ' (GOAL)'
                        ax.text(x+.05, ys-y-0.95, txt)

                    if state == self.goal_state:
                        continue

                    tx, ty = x, ys-y-1

                    action_map = {
                        0: ((0.5+tx, 0.5+ty), (tx+1, ty+1), (tx, ty+1)),
                        1: ((tx, ty), (tx+1, ty), (tx+0.5, ty+0.5)),
                        2: ((tx, ty), (tx+0.5, ty+0.5), (tx, ty+1)),
                        3: ((0.5+tx, 0.5+ty), (tx+1, ty), (tx+1, ty+1)),
                    }
                    offset_map = {
                        0: (0.1, 0.8),
                        1: (0.1, 0.1),
                        2: (-0.2, 0.4),
                        3: (0.4, 0.4),
                    }

                    if state in self.wall_states:
                        ax.add_patch(plt.Rectangle((tx, ty), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
                        continue

                    for action in action_space:
                        if state == self.goal_state:
                            continue

                        tq = q.get((state, action), 0)
                        color_scale = 0.5 + (tq / qmax) / 2  # normalize: 0.0-1.0

                        poly = plt.Polygon(action_map[action], fc=cmap(color_scale))
                        ax.add_patch(poly)

                        if print_value:
                            offset = offset_map[action]
                            ax.text(tx+offset[0], ty+offset[1], "{:12.2f}".format(tq))

        plt.show()

        if q is not None:
            policy = {}
            for y in range(ys):
                for x in range(xs):
                    state = (y, x)
                    if state not in self.wall_states:
                        qs = [q.get((state, action), 0) for action in range(4)]
                        max_action = np.argmax(qs)
                        probs = {a: 0.0 for a in range(4)}
                        probs[max_action] = 1.0
                        policy[state] = probs
            self.render_v(None, policy)