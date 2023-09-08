import gymnasium as gym
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import json

MAX_SCORE = 3932156
MAX_CELL_VALUE = 65536

class GameEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            uri):
        super().__init__()
        self.uri = uri
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=MAX_CELL_VALUE, 
            shape=(16,), 
            dtype=np.uint8
        )

        self.prev_score = 0

        _options = webdriver.ChromeOptions()
        _options.add_argument("--mute-audio")

        self._driver = webdriver.Chrome(
            options=_options
        )

        self.actions_map = [
            Keys.ARROW_UP,
            Keys.ARROW_RIGHT,
            Keys.ARROW_DOWN,
            Keys.ARROW_LEFT
        ]

        self.to_arrow = {
            0: '⇧',
            1: '⇨',
            2: '⇩',
            3: '⇦'
        }
    
    def _get_record(self):
        record = self._driver.execute_script('return localStorage.getItem("bestScore");')
        return int(record) if record else 0
    
    def _get_state(self):
        gameState = self._driver.execute_script('return localStorage.getItem("gameState");')
        cells = np.zeros(16)
        #score = 0
        if gameState:
            gameState = json.loads(gameState)
            #score = gameState['score']
            cells = np.array([d.get('value', 0) if d else 0 \
                                         for row in gameState['grid']['cells'] for d in row])
        #record = self._get_record()
        return gameState, cells #np.array([score, record] + cells.tolist())

    def step(self, action):
        self._driver.find_element(By.TAG_NAME, 'body') \
            .send_keys(self.actions_map[action])
        
        gameState, obs = self._get_state()
        terminated = gameState == None

        reward = -1
        truncated = False
        score = 0

        if not terminated:
            score = gameState['score']
            gained_score = score - self.prev_score

            if gained_score != 0:
                reward = score
            else:
                reward = -.5
            
            self.prev_score = score

        arrow = self.to_arrow[action]

        self._update_analytics(arrow, reward)

        return obs, reward, terminated, truncated, {'score': score}
    
    def _next_observation(self):
        _, obs = self._get_state()
        return obs
    
    def _inject_analytics(self):
        js_code = """
            var existingParagraph = document.querySelector('.above-game p');
            var newParagraph = document.createElement('p');
            newParagraph.className = 'analytics';
            existingParagraph.parentNode.insertBefore(newParagraph, existingParagraph.nextSibling);
        """
        self._driver.execute_script(js_code)
    
    def _update_analytics(self, action, reward):
        #reward = "{:.10f}".format(reward)
        js_code = f"""
            var p = document.querySelector('.analytics');
            p.textContent = 'Action: {action} Reward: {reward}';
        """
        self._driver.execute_script(js_code)

    def reset(self, seed=None, options=None):
        self.prev_score = 0
        self._driver.get(self.uri)
        self._inject_analytics()
        return self._next_observation(), {}

    def render(self, mode: str='human'):
        return

    def close(self):
        return
