"""
F1 Race Road Game AI Environment (cleaned)
=========================================

This module defines a simple Gym-like environment that exposes:
- reset() -> state: np.ndarray
- step(action: int) -> (next_state, reward, done, info)
- render() -> None
- close() -> None

It mirrors the behavior in `f1_race_env.py` but lives under `src/` for a
cleaner repository layout and import path.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, Tuple

import numpy as np
import pygame


class F1RaceEnvironment:
	"""
	Simple obstacle-dodging environment for an AI-controlled car.

	Actions: 0=stay, 1=left, 2=right
	State (5 floats, each ~[0,1]):
		[car_x_norm, obstacle_x_norm, obstacle_closeness, speed_ratio, distance_ratio]
	"""

	def __init__(self, render: bool = True) -> None:
		pygame.init()

		# Display
		self.SCREEN_WIDTH = 400
		self.SCREEN_HEIGHT = 600
		self.render_game = render
		if self.render_game:
			self.game_display = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
			pygame.display.set_caption("F1 Race AI Training")
		else:
			# Headless surface for fast training
			self.game_display = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
		self.clock = pygame.time.Clock()

		# Colors
		self.BLACK = (0, 0, 0)
		self.WHITE = (255, 255, 255)
		self.RED = (255, 0, 0)
		self.GREEN = (0, 255, 0)
		self.BLUE = (0, 0, 255)
		self.GRAY = (64, 64, 64)

		# Dynamics
		self.CAR_SPEED = 8
		self.INITIAL_OBSTACLE_SPEED = 5
		self.SPEED_INCREASE = 0.5
		self.MAX_SPEED = 20

		# Rewards
		self.CRASH_PENALTY = -100.0
		self.SURVIVAL_REWARD = 0.1
		self.DODGE_BONUS = 10.0
		self.MOVEMENT_COST = -0.01

		# Load images
		self._load_images()

		# Action/state spaces
		self.action_space_size = 3
		self.state_space_size = 5

		# Start
		self.reset()

	def _asset(self, *parts: str) -> str:
		return os.path.join("assets", *parts).replace("\\", "/")

	def _load_images(self) -> None:
		try:
			self.car_img = pygame.image.load(self._asset("images", "car.png"))
			self.car_left_img = pygame.image.load(self._asset("images", "car_left.png"))
			self.car_right_img = pygame.image.load(self._asset("images", "car_right.png"))
			self.obstacle_img = pygame.image.load(self._asset("images", "obstacle.png"))
			self.background_img = pygame.image.load(self._asset("images", "background.png"))
			self.texture_img = pygame.image.load(self._asset("images", "texture.png"))
		except pygame.error:
			# Fallback to colored rectangles if assets not present
			self.car_img = pygame.Surface((40, 60))
			self.car_img.fill(self.BLUE)
			self.car_left_img = self.car_img.copy()
			self.car_right_img = self.car_img.copy()
			self.obstacle_img = pygame.Surface((40, 60))
			self.obstacle_img.fill(self.RED)
			self.background_img = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
			self.background_img.fill(self.BLACK)
			self.texture_img = pygame.Surface((self.SCREEN_WIDTH, 400))
			self.texture_img.fill(self.GRAY)

		car_rect = self.car_img.get_rect()
		self.car_width, self.car_height = car_rect.width, car_rect.height
		obs_rect = self.obstacle_img.get_rect()
		self.obstacle_width, self.obstacle_height = obs_rect.width, obs_rect.height

	def reset(self) -> np.ndarray:
		# Car
		self.car_x = self.SCREEN_WIDTH * 0.4
		self.car_y = self.SCREEN_HEIGHT * 0.8
		self.car_direction = 0  # -1 left, 0 straight, 1 right

		# Obstacle
		safe_left = 8
		safe_right = self.SCREEN_WIDTH - self.obstacle_width - 8
		self.obstacle_x = random.randrange(safe_left, safe_right)
		self.obstacle_y = -600
		self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED

		# Stats
		self.score = 0
		self.game_over = False
		self.frames_survived = 0
		self.texture_y = 0

		return self._state()

	def _state(self) -> np.ndarray:
		car_position = self.car_x / self.SCREEN_WIDTH
		obstacle_position = self.obstacle_x / self.SCREEN_WIDTH

		total_possible_distance = self.SCREEN_HEIGHT + 600
		obstacle_closeness = (self.obstacle_y + 600) / total_possible_distance

		speed_ratio = min(self.obstacle_speed / self.MAX_SPEED, 1.0)

		horizontal_distance = abs(self.car_x - self.obstacle_x)
		vertical_distance = abs(self.car_y - self.obstacle_y)
		total_distance = math.hypot(horizontal_distance, vertical_distance)
		max_distance = math.hypot(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
		distance_ratio = total_distance / max_distance

		return np.array(
			[car_position, obstacle_position, obstacle_closeness, speed_ratio, distance_ratio],
			dtype=np.float32,
		)

	def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
		if self.game_over:
			return self._state(), 0.0, True, {}

		# Handle window close
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.game_over = True

		self._move_car(action)
		self._update_world()
		reward = self._reward()

		next_state = self._state()
		done = self.game_over
		info = {"score": self.score, "frames": self.frames_survived, "speed": self.obstacle_speed}

		self.frames_survived += 1
		return next_state, float(reward), bool(done), info

	def _move_car(self, action: int) -> None:
		if action == 1:
			self.car_x -= self.CAR_SPEED
			self.car_direction = -1
			if self.car_x < 0:
				self.car_x = 0
		elif action == 2:
			self.car_x += self.CAR_SPEED
			self.car_direction = 1
			max_x = self.SCREEN_WIDTH - self.car_width
			if self.car_x > max_x:
				self.car_x = max_x
		else:
			self.car_direction = 0

	def _update_world(self) -> None:
		self.obstacle_y += self.obstacle_speed
		self.texture_y += self.obstacle_speed
		if self.texture_y >= 400:
			self.texture_y = 0

		# Passed obstacle
		if self.obstacle_y > self.SCREEN_HEIGHT:
			self.score += 1
			safe_left = 8
			safe_right = self.SCREEN_WIDTH - self.obstacle_width - 8
			self.obstacle_x = random.randrange(safe_left, safe_right)
			self.obstacle_y = -self.obstacle_height
			self.obstacle_speed = min(self.obstacle_speed + self.SPEED_INCREASE, self.MAX_SPEED)

		# Collisions or bounds
		if self._collided():
			self.game_over = True
		if self.car_x < 0 or self.car_x > (self.SCREEN_WIDTH - self.car_width):
			self.game_over = True

	def _collided(self) -> bool:
		car_rect = pygame.Rect(self.car_x, self.car_y, self.car_width, self.car_height)
		obs_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)
		return car_rect.colliderect(obs_rect)

	def _reward(self) -> float:
		if self.game_over:
			return self.CRASH_PENALTY
		reward = self.SURVIVAL_REWARD
		just_passed = self.obstacle_y > self.car_y and self.obstacle_y <= self.car_y + self.obstacle_speed
		if just_passed:
			reward += self.DODGE_BONUS
		if self.car_direction != 0:
			reward += self.MOVEMENT_COST
		return reward

	def render(self) -> None:
		if not self.render_game:
			return
		self.game_display.blit(self.background_img, (0, 0))
		self.game_display.blit(self.texture_img, (0, self.texture_y - 400))
		self.game_display.blit(self.texture_img, (0, self.texture_y))
		self.game_display.blit(self.texture_img, (0, self.texture_y + 400))
		self.game_display.blit(self.obstacle_img, (self.obstacle_x, self.obstacle_y))
		car_img = self.car_img if self.car_direction == 0 else (self.car_left_img if self.car_direction == -1 else self.car_right_img)
		self.game_display.blit(car_img, (self.car_x, self.car_y))

		font = pygame.font.SysFont(None, 25)
		self.game_display.blit(font.render(f"Score: {self.score}", True, self.GREEN), (10, 10))
		self.game_display.blit(font.render(f"Speed: {self.obstacle_speed:.1f}", True, self.GREEN), (10, 35))
		self.game_display.blit(font.render(f"Frames: {self.frames_survived}", True, self.GREEN), (10, 60))
		pygame.display.update()

	def close(self) -> None:
		pygame.quit()

