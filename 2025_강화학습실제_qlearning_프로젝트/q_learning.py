from collections import defaultdict
import numpy as np
from gridworld import GridWorld
import gridworld_render as render_helper
import matplotlib.pyplot as plt
import matplotlib


class FourGridWorld(GridWorld):
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        
        self.reward_map = np.array([
            [0, 0, 0, 0],
            [0, None, 0, None],
            [0, 0, 0, None],
            [None, 0, 0, 1]
        ])
        
        self.goal_state = (3, 3)
        self.wall_states = [(1, 1), (1, 3), (2, 3), (3, 0)]
        self.start_state = (0, 0)
        self.agent_state = self.start_state
        self.dynamic_walls = []
        self.max_steps = 100  # 최대 스텝 수 제한

    def add_dynamic_wall(self, position):
        """동적으로 장애물을 추가하는 메서드"""
        if position not in self.wall_states and position not in self.dynamic_walls:
            self.dynamic_walls.append(position)
            return True
        return False

    def remove_dynamic_wall(self, position):
        """동적 장애물을 제거하는 메서드"""
        if position in self.dynamic_walls:
            self.dynamic_walls.remove(position)
            return True
        return False

    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state in self.wall_states:
            next_state = state
        # 1->2 경로 차단 (0,0)->(0,1)
        elif state == (0, 0) and next_state == (0, 1) and (0, 1) in self.dynamic_walls:
            next_state = state
        # 11->15 경로 차단 (2,2)->(3,2)
        elif state == (2, 2) and next_state == (3, 2) and (3, 2) in self.dynamic_walls:
            next_state = state

        return next_state
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                state = (h, w)
                if state not in self.wall_states:
                    yield state
    def render_v(self, v=None, policy=None, print_value=True, path=None):
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
        
        # 경로 시각화
        if path is not None:
            for i in range(len(path)-1):
                current = path[i]
                next_state = path[i+1]
                # 경로를 선으로 연결
                ax.plot([current[1]+0.5, next_state[1]+0.5], 
                        [ys-current[0]-0.5, ys-next_state[0]-0.5], 
                        'r-', linewidth=3)
                # 현재 위치를 점으로 표시
                ax.plot(current[1]+0.5, ys-current[0]-0.5, 'ro', markersize=10)
            # 목표 위치를 점으로 표시
            ax.plot(path[-1][1]+0.5, ys-path[-1][0]-0.5, 'go', markersize=10)
        
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
                    
                    # arrows = ["↑", "↓", "←", "→"]
                    # offsets = [(0, 0.1), (0, -0.1), (-0.1, 0), (0.1, 0)]
                    for action in max_actions:
                        # arrow = arrows[action]
                        # offset = offsets[action]
                        if state == self.goal_state:
                            continue
                        # ax.text(x+0.45+offset[0], ys-y-0.5+offset[1], arrow)
                
                if state in self.wall_states:
                    ax.add_patch(plt.Rectangle((x,ys-y-1), 1, 1, fc=(0.4, 0.4, 0.4, 1.)))
                if state in self.dynamic_walls:
                    ax.add_patch(plt.Rectangle((x,ys-y-1), 1, 1, fc=(0.8, 0.2, 0.2, 0.5)))  # 동적 장애물은 빨간색으로 표시
        
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

    def get_best_action(self, state, Q):
        """현재 상태에서 가장 높은 Q값을 가진 행동 선택"""
        # 각 행동의 Q값을 저장
        q_values = []
        available_actions = []
        
        for action in self.action_space:
            next_state = self.next_state(state, action)
            if next_state != state:  # 이동 가능한 경우만
                q_values.append(Q.get((state, action), 0))
                available_actions.append(action)
        
        if not available_actions:
            return None
            
        # Q값이 높은 순서대로 행동 정렬
        sorted_indices = np.argsort(q_values)[::-1]  # 내림차순
        return available_actions[sorted_indices[0]]  # 가장 높은 Q값의 행동 반환

    def has_obstacle_in_direction(self, state, action):
        """현재 방향에 장애물이 있는지 확인"""
        next_state = self.next_state(state, action)
        return next_state == state  # 움직일 수 없다면 장애물이 있는 것

    def find_optimal_direction(self, state, Q, excluded_actions=None):
        """Q값을 기반으로 최적의 방향 찾기"""
        available_actions = []
        q_values = []
        
        # 먼저 각 방향이 실제로 이동 가능한지 확인
        for action in self.action_space:
            if excluded_actions and action in excluded_actions:
                continue
            
            next_state = self.next_state(state, action)
            if next_state != state:  # 이동 가능한 경우만 고려
                available_actions.append(action)
                q_values.append(Q.get((state, action), 0))
        
        if not available_actions:
            return None
        
        # Q값이 가장 높은 행동 선택
        max_idx = np.argmax(q_values)
        return available_actions[max_idx]

    def find_path_with_dynamic_walls(self, Q):
        """동적 장애물을 피해 경로 찾기"""
        path = []
        current_state = self.start_state
        visited = set()  # 방문한 상태 기록
        
        while current_state != self.goal_state:
            if current_state in visited:  # 순환 경로 방지
                return None
            
            visited.add(current_state)
            path.append(current_state)
            
            # 현재 상태에서 가능한 모든 행동의 Q값 계산
            q_values = []
            available_actions = []
            
            for action in self.action_space:
                next_state = self.next_state(current_state, action)
                if next_state != current_state:  # 이동 가능한 경우만
                    q_values.append(Q.get((current_state, action), 0))
                    available_actions.append(action)
            
            if not available_actions:
                return None  # 더 이상 이동할 수 없음
            
            # Q값이 높은 순서대로 행동 정렬
            sorted_indices = np.argsort(q_values)[::-1]
            
            # Q값이 높은 행동부터 시도
            moved = False
            for idx in sorted_indices:
                action = available_actions[idx]
                next_state = self.next_state(current_state, action)
                
                if next_state != current_state and next_state not in visited:
                    current_state = next_state
                    moved = True
                    break
            
            if not moved:
                return None  # 더 이상 진행할 수 없음
        
        path.append(self.goal_state)
        return path

    def step(self, action):
        self.agent_state = self.next_state(self.agent_state, action)
        reward = 0
        done = False

        if self.agent_state == self.goal_state:
            reward = 1
            done = True
        elif self.agent_state in self.wall_states or self.agent_state in self.dynamic_walls:
            reward = -1
            done = True
        else:
            reward = -0.01  # 매 스텝마다 작은 페널티

        return self.agent_state, reward, done

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8  # 학습률을 다시 0.8로
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 0.0001
        self.epsilon = self.epsilon_start
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            max_q = max(qs)
            # 최대 Q값을 가진 행동들을 모두 찾아서 그 중에서 랜덤하게 선택
            max_actions = [a for a, q in enumerate(qs) if q == max_q]
            return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
        # 입실론 값 감소
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * (1 - self.epsilon_decay)  # 지수적 감소로 변경
        )

def get_v(Q, env):
    V = defaultdict(float)
    for state in env.states():
        qs = [Q[(state, action)] for action in range(4)]
        V[state] = max(qs)
    return V

def get_policy(Q, env):
    policy = {}
    for state in env.states():
        qs = [Q[(state, action)] for action in range(4)]
        max_action = np.argmax(qs)
        probs = {a: 0.0 for a in range(4)}
        probs[max_action] = 1.0
        policy[state] = probs
    return policy

if __name__ == '__main__':
    env = FourGridWorld()
    agent = QLearningAgent()

    # 기본 그리드월드 시각화 (장애물만)
    print("\n=== 기본 그리드월드 ===")
    env.render_v()

    # Q-table 학습
    episodes = 10000
    epsilon_history = []
    steps_history = []
    rewards_history = []
    
    print("=== Q-learning 학습 시작 ===")
    for episode in range(episodes):
        state = env.reset()
        epsilon_history.append(agent.epsilon)
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            steps += 1

            if done or steps >= env.max_steps:  # 최대 스텝 수 제한 추가
                steps_history.append(steps)
                rewards_history.append(total_reward)
                break
            state = next_state
        
        # 입실론 감소
        agent.epsilon = max(
            agent.epsilon_end,
            agent.epsilon * (1 - agent.epsilon_decay)  # 지수적 감소로 변경
        )
    
    print("\n=== 학습 결과 ===")
    print(f"평균 스텝 수: {sum(steps_history[-1000:]) / 1000:.2f}")
    print(f"평균 보상: {sum(rewards_history[-1000:]) / 1000:.2f}")
    print(f"최소 스텝 수: {min(steps_history)}")
    
    # 학습 과정 시각화
    plt.figure(figsize=(15, 5))
    
    # 스텝 수 변화
    plt.subplot(1, 3, 1)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # 보상 변화
    plt.subplot(1, 3, 2)
    plt.plot(rewards_history)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # 입실론 변화
    plt.subplot(1, 3, 3)
    plt.plot(epsilon_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.show()
    # 학습된 Q-table로 기본 경로 찾기
    V = get_v(agent.Q, env)
    policy = get_policy(agent.Q, env)
    original_path = env.find_path_with_dynamic_walls(agent.Q)
    env.render_q(agent.Q)
    if original_path:
        print("\n=== 기본 경로 ===")
        print(f"경로 길이: {len(original_path)}")
        print("경로:", original_path)
        env.render_v(V, policy, path=original_path)
    
    # Case 1: 1->2 경로가 막힐 때
    print("\n=== Case 1: 1->2 경로가 막힐 때 ===")
    env.dynamic_walls = []  # 이전 장애물 모두 제거
    env.add_dynamic_wall((0, 1))
    print("(0,0)->(0,1) 경로가 막힘")
    
    new_path = env.find_path_with_dynamic_walls(agent.Q)
    if new_path:
        print(f"새로운 경로 길이: {len(new_path)}")
        print("새로운 경로:", new_path)
        env.render_v(V, policy, path=new_path)
    else:
        print("경로를 찾을 수 없습니다.")
    
    # Case 2: 11->15 경로가 막힐 때
    print("\n=== Case 2: 11->15 경로가 막힐 때 ===")
    env.dynamic_walls = []  # 이전 장애물 모두 제거
    env.add_dynamic_wall((3, 2))
    print("(2,2)->(3,2) 경로가 막힘")
    
    new_path = env.find_path_with_dynamic_walls(agent.Q)
    if new_path:
        print(f"새로운 경로 길이: {len(new_path)}")
        print("새로운 경로:", new_path)
        env.render_v(V, policy, path=new_path)
    else:
        print("경로를 찾을 수 없습니다.") 