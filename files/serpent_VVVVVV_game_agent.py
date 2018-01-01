import serpent.cv
import serpent.utilities
import serpent.ocr

from serpent.game_agent import GameAgent
from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey
from serpent.sprite import Sprite
from serpent.sprite_locator import SpriteLocator

from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace

from datetime import datetime

import numpy as np
from PIL import Image

import skimage.color
import skimage.measure

import os
import time
import gc
import collections


class SerpentVVVVVVGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None
        self._reset_game_state()

    def setup_play(self):
        input_mapping = {
            #"UP": [KeyboardKey.KEY_UP],
            "LEFT": [KeyboardKey.KEY_LEFT],
            #"DOWN": [KeyboardKey.KEY_DOWN],
            "RIGHT": [KeyboardKey.KEY_RIGHT]
        }

        self.key_mapping = {
            #KeyboardKey.KEY_UP.name: "UP",
            KeyboardKey.KEY_LEFT.name: "LEFT",
            #KeyboardKey.KEY_DOWN.name: "DOWN",
            KeyboardKey.KEY_RIGHT.name: "RIGHT"
        }

        direction_action_space = KeyboardMouseActionSpace(
            direction_keys=[None, "LEFT", "RIGHT"]
        )

        direction_model_file_path = "datasets/vvvvvv_direction_dqn_0_1_.h5".replace("/", os.sep)

        self.dqn_direction = DDQN(
            model_file_path=direction_model_file_path if os.path.isfile(direction_model_file_path) else None,
            input_shape=(480, 640, 4),
            input_mapping=input_mapping,
            action_space=direction_action_space,
            replay_memory_size=5000,
            max_steps=1000000,
            observe_steps=600,
            batch_size=32,
            model_learning_rate=1e-4,
            initial_epsilon=1,
            final_epsilon=0.01,
            override_epsilon=False
        )

    def handle_play(self, game_frame):
        gc.disable()

        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )

        if self.dqn_direction.first_run:
            #self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
            #time.sleep(5)

            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

            self.dqn_direction.first_run = False

            time.sleep(5)

            return None

        actor_hp = self._measure_actor_hp(game_frame)
        run_score = self._measure_run_score(game_frame)

        self.game_state["health"].appendleft(actor_hp)
        self.game_state["score"].appendleft(run_score)

        if self.dqn_direction.frame_stack is None:
            full_game_frame = FrameGrabber.get_frames(
                [0],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE"
            ).frames[0]

            self.dqn_direction.build_frame_stack(full_game_frame.frame)
        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE"
            )

            if self.dqn_direction.mode == "TRAIN":
                reward_direction, reward_action = self._calculate_reward()

                self.game_state["run_reward_direction"] += reward_direction
                self.game_state["run_reward_action"] += reward_action

                self.dqn_direction.append_to_replay_memory(
                    game_frame_buffer,
                    reward_direction,
                    terminal=self.game_state["health"] == 0
                )

                # Every 2000 steps, save latest weights to disk
                if self.dqn_direction.current_step % 2000 == 0:
                    self.dqn_direction.save_model_weights(
                        file_path_prefix=f"datasets/vvvvvv_direction"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.dqn_direction.current_step % 20000 == 0:
                    self.dqn_direction.save_model_weights(
                        file_path_prefix=f"datasets/vvvvvv_direction",
                        is_checkpoint=True
                    )

            elif self.dqn_direction.mode == "RUN":
                self.dqn_direction.update_frame_stack(game_frame_buffer)

            run_time = datetime.now() - self.started_at

            serpent.utilities.clear_terminal()

            print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
            print("GAME: VVVVVV                 PLATFORM: Steam                AGENT: DDQN + Prioritized Experience Replay")
            print("")

            self.dqn_direction.output_step_data()

            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_direction'] + self.game_state['run_reward_action'], 2)}")
            print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            print(f"CURRENT HEALTH: {self.game_state['health'][0]}")
            print(f"CURRENT SCORE: {self.game_state['score'][0]}")
            print("")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")

            print("")
            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'})")
            print("")

            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")

            if self.game_state["health"][1] <= 0:
                serpent.utilities.clear_terminal()
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.dqn_direction.mode in ["TRAIN", "RUN"]:
                    # Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_direction.mode == "RUN"
                        }
                else:
                    self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
                    self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])

                self.game_state["current_run_steps"] = 0

                self.input_controller.handle_keys([])

                if self.dqn_direction.mode == "TRAIN":
                    for i in range(8):
                        run_time = datetime.now() - self.started_at
                        serpent.utilities.clear_terminal()
                        print(f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds")
                        print("GAME: VVVVVV                 PLATFORM: Steam                AGENT: DDQN + Prioritized Experience Replay")
                        print("")

                        print(f"TRAINING ON MINI-BATCHES: {i + 1}/2")
                        print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                        self.dqn_direction.train_on_mini_batch()

                self.game_state["run_timestamp"] = datetime.utcnow()
                self.game_state["current_run"] += 1
                self.game_state["run_reward_direction"] = 0
                self.game_state["run_reward_action"] = 0
                self.game_state["run_predicted_actions"] = 0
                self.game_state["health"] = collections.deque(np.full((8,), 3), maxlen=8)
                self.game_state["score"] = collections.deque(np.full((8,), 0), maxlen=8)

                if self.dqn_direction.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        if self.dqn_direction.type == "DDQN":
                            self.dqn_direction.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.dqn_direction.enter_run_mode()
                    else:
                        self.dqn_direction.enter_train_mode()

                #self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                #time.sleep(3)

                self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

                return None

        self.dqn_direction.pick_action()
        self.dqn_direction.generate_action()

        keys = self.dqn_direction.get_input_values()
        print("")

        print("PRESSING: ", end='')
        print(" + ".join(list(map(lambda k: self.key_mapping.get(k.name), keys))))

        self.input_controller.handle_keys(keys)

        if self.dqn_direction.current_action_type == "PREDICTED":
            self.game_state["run_predicted_actions"] += 1

        self.dqn_direction.erode_epsilon(factor=2)

        self.dqn_direction.next_step()

        self.game_state["current_run_steps"] += 1

    def _reset_game_state(self):
        self.game_state = {
            "health": collections.deque(np.full((8,), 3), maxlen=8),
            "score": collections.deque(np.full((8,), 0), maxlen=8),
            "run_reward_direction": 0,
            "run_reward_action": 0,
            "current_run": 1,
            "current_run_steps": 0,
            "current_run_health": 0,
            "current_run_score": 0,
            "run_predicted_actions": 0,
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "run_timestamp": datetime.utcnow(),
        }

    def _measure_actor_hp(self, game_frame):
        hp_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["HP_AREA"])
        hp_area_image = Image.fromarray(hp_area_frame)

        actor_hp = 0

        image_colors = hp_area_image.getcolors() # TODO: remove in favor of sprite detection and location
        if image_colors:
            actor_hp = len(image_colors) - 7

        for name, sprite in self.game.sprites.items():
            query_sprite = Sprite("QUERY", image_data=sprite.image_data)
            sprite_name = self.sprite_identifier.identify(query_sprite, mode="CONSTELLATION_OF_PIXELS")  # Will be "UNKNOWN" if no match
            print(sprite_name)
            sprite_to_locate = Sprite("QUERY", image_data=sprite.image_data)

            sprite_locator = SpriteLocator()
            location = sprite_locator.locate(sprite=sprite_to_locate, game_frame=game_frame)
            print(location)
            if location:
                actor_hp = 1000000

        return actor_hp

    def _measure_run_score(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["SCORE_AREA"])

        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
        score_image = Image.fromarray(score_grayscale)

        score = '0'

        image_colors = score_image.getcolors()
        if image_colors and len(image_colors) > 1:
            score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5)
            score = score.split(":")[0]

        count = 0

        if not score.isdigit():
            score = '0'

        self.game_state["current_run_score"] = score

        return score

    def _calculate_reward(self):
        reward = 0

        reward += self.game_state["health"][0] / 10.0
        reward += (0.5 if (int(self.game_state["score"][0]) - int(self.game_state["score"][1])) >= 0 else -0.25)

        return reward, reward
