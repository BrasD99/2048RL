import gymnasium as gym
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from io import BytesIO
from PIL import Image
import base64
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
            low=1, 
            high=1, 
            shape=(18,), 
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

        action_chains = ActionChains(self._driver)
        self.keydown_actions = [action_chains.key_down(item) for item in self.actions_map]
        self.keyup_actions = [action_chains.key_up(item) for item in self.actions_map]

        self.map_actions = {
            0: '⇧',
            1: '⇨',
            2: '⇩',
            3: '⇦'
        }
        
    def _get_done(self):
        game_message_element = self._driver \
            .find_element(By.CLASS_NAME, 'game-message')
        class_attribute = game_message_element \
            .get_attribute('class')
        return 'game-over' in class_attribute
    
    def _get_image(self):
        _img = self._driver \
            .find_element(By.CLASS_NAME, 'game-container')
        return np.array(
            Image.open(BytesIO(base64.b64decode(_img.screenshot_as_base64)))
        )
    
    def _get_record(self):
        record = self._driver.execute_script('return localStorage.getItem("bestScore");')
        return int(record) if record else 0
    
    def _get_state(self):
        gameState = self._driver.execute_script('return localStorage.getItem("gameState");')
        cells = np.zeros(16)
        score = 0
        if gameState:
            gameState = json.loads(gameState)
            score = gameState['score']
            cells = np.array([d.get('value', 0) if d else 0 \
                                         for row in gameState['grid']['cells'] for d in row])
        record = self._get_record()
        return gameState, np.array([score, record] + cells.tolist())

    def step(self, action):
        self._driver.find_element(By.TAG_NAME, 'body') \
            .send_keys(self.actions_map[action])
        
        gameState, obs = self._get_state()
        terminated = gameState == None

        reward = -1
        truncated = False
        score = 0
        gained_score = 0

        if not terminated:
            score = gameState['score']
            gained_score = score - self.prev_score
            if gained_score == 0:
                reward = -1
            else:
                reward = gained_score
                
            self.prev_score = score

        action = self.map_actions[action]

        self._update_analytics(action, reward)

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
        reward = "{:.10f}".format(reward)
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
