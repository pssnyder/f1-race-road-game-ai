import pygame
import random
import numpy as np
import os
import math

class F1RaceEnvironment:
    """
    F1 Race Game Environment for AI Training
    Modified version that allows programmatic control and state extraction
    """
    
    def __init__(self, render=True):
        pygame.init()
        
        # Game constants
        self.screen_width = 400
        self.screen_height = 600
        self.render_game = render
        
        # Colors
        self.black_color = (0, 0, 0)
        self.white_color = (255, 255, 255)
        self.red_color = (255, 0, 0)
        self.green_color = (0, 255, 0)
        self.blue_color = (0, 0, 255)
        
        # Initialize display (only if rendering)
        if self.render_game:
            self.game_display = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('F1 Race AI Training')
        else:
            # Create a dummy surface for non-rendering mode
            self.game_display = pygame.Surface((self.screen_width, self.screen_height))
            
        self.clock = pygame.time.Clock()
        
        # Load images
        try:
            self.car_img = pygame.image.load('images/car.png')
            self.car_left_img = pygame.image.load('images/car_left.png')
            self.car_right_img = pygame.image.load('images/car_right.png')
            self.obstacle_img = pygame.image.load('images/obstacle.png')
            self.background_img = pygame.image.load('images/background.png')
            self.texture_img = pygame.image.load('images/texture.png')
        except pygame.error as e:
            print(f"Error loading images: {e}")
            # Create dummy colored rectangles if images fail to load
            self.car_img = pygame.Surface((40, 60))
            self.car_img.fill(self.blue_color)
            self.car_left_img = self.car_img.copy()
            self.car_right_img = self.car_img.copy()
            self.obstacle_img = pygame.Surface((40, 60))
            self.obstacle_img.fill(self.red_color)
            self.background_img = pygame.Surface((self.screen_width, self.screen_height))
            self.background_img.fill(self.black_color)
            self.texture_img = pygame.Surface((self.screen_width, 400))
            self.texture_img.fill((64, 64, 64))
        
        # Get image dimensions
        self.car_width, self.car_height = self.car_img.get_rect().size
        self.obstacle_width, self.obstacle_height = self.obstacle_img.get_rect().size
        
        # Initialize game state
        self.reset()
        
        # Action space: 0=no action, 1=left, 2=right
        self.action_space_size = 3
        
        # State space: [car_x_normalized, obstacle_x_normalized, obstacle_y_normalized, 
        #               speed_normalized, distance_to_obstacle_normalized]
        self.state_space_size = 5
        
    def reset(self):
        """Reset the game to initial state"""
        # Car position
        self.car_x = self.screen_width * 0.4  # Start in middle
        self.car_y = self.screen_height * 0.8  # Near bottom
        self.car_direction = 0  # -1=left, 0=center, 1=right
        
        # Obstacle position
        self.obstacle_x = random.randrange(8, self.screen_width - self.obstacle_width - 8)
        self.obstacle_y = -600  # Start above screen
        self.obstacle_speed = 5
        
        # Game state
        self.score = 0
        self.game_over = False
        self.frames_survived = 0
        self.texture_y = 0
        
        return self._get_state()
    
    def _get_state(self):
        """Extract current game state as normalized vector"""
        # Normalize all values to [0, 1] range
        car_x_norm = self.car_x / self.screen_width
        obstacle_x_norm = self.obstacle_x / self.screen_width  
        obstacle_y_norm = (self.obstacle_y + 600) / (self.screen_height + 600)  # Normalize from -600 to screen_height
        speed_norm = min(self.obstacle_speed / 20.0, 1.0)  # Assume max speed of 20
        
        # Calculate distance to obstacle
        distance = math.sqrt((self.car_x - self.obstacle_x)**2 + (self.car_y - self.obstacle_y)**2)
        max_distance = math.sqrt(self.screen_width**2 + self.screen_height**2)
        distance_norm = distance / max_distance
        
        return np.array([car_x_norm, obstacle_x_norm, obstacle_y_norm, speed_norm, distance_norm], dtype=np.float32)
    
    def step(self, action):
        """
        Execute one game step with given action
        Returns: next_state, reward, done, info
        """
        if self.game_over:
            return self._get_state(), 0, True, {}
        
        # Handle pygame events (but ignore them for AI control)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
        
        # Apply action
        self._apply_action(action)
        
        # Update game state
        self._update_game()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if game is over
        done = self.game_over
        
        # Get next state
        next_state = self._get_state()
        
        self.frames_survived += 1
        
        return next_state, reward, done, {"score": self.score, "frames": self.frames_survived}
    
    def _apply_action(self, action):
        """Apply the given action to the car"""
        move_speed = 8
        
        if action == 1:  # Move left
            self.car_x -= move_speed
            self.car_direction = -1
            # Prevent going off screen
            if self.car_x < 0:
                self.car_x = 0
        elif action == 2:  # Move right
            self.car_x += move_speed
            self.car_direction = 1
            # Prevent going off screen
            if self.car_x > self.screen_width - self.car_width:
                self.car_x = self.screen_width - self.car_width
        else:  # No action
            self.car_direction = 0
    
    def _update_game(self):
        """Update game state (obstacle movement, collision detection, etc.)"""
        # Move obstacle down
        self.obstacle_y += self.obstacle_speed
        
        # Move texture for visual effect
        self.texture_y += self.obstacle_speed
        if self.texture_y >= 400:
            self.texture_y = 0
        
        # Check if obstacle passed the car (score point)
        if self.obstacle_y > self.screen_height:
            self.score += 1
            self.obstacle_y = -self.obstacle_height
            self.obstacle_x = random.randrange(8, self.screen_width - self.obstacle_width - 8)
            self.obstacle_speed += 0.5  # Increase speed gradually
        
        # Check collision
        if self._check_collision():
            self.game_over = True
        
        # Check if car went off screen (also game over)
        if self.car_x < 0 or self.car_x > self.screen_width - self.car_width:
            self.game_over = True
    
    def _check_collision(self):
        """Check if car collides with obstacle"""
        car_rect = pygame.Rect(self.car_x, self.car_y, self.car_width, self.car_height)
        obstacle_rect = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)
        return car_rect.colliderect(obstacle_rect)
    
    def _calculate_reward(self):
        """Calculate reward for current state"""
        if self.game_over:
            return -100  # Large penalty for crashing
        
        reward = 0.1  # Small reward for surviving each frame
        
        # Bonus for dodging obstacle (when it passes the car)
        if self.obstacle_y > self.car_y and self.obstacle_y <= self.car_y + self.obstacle_speed:
            reward += 10
        
        # Small penalty for unnecessary movement (encourage efficiency)
        if self.car_direction != 0:
            reward -= 0.01
            
        return reward
    
    def render(self):
        """Render the game (only if render_game=True)"""
        if not self.render_game:
            return
            
        # Clear screen
        self.game_display.blit(self.background_img, (0, 0))
        
        # Draw moving texture
        self.game_display.blit(self.texture_img, (0, self.texture_y - 400))
        self.game_display.blit(self.texture_img, (0, self.texture_y))
        self.game_display.blit(self.texture_img, (0, self.texture_y + 400))
        
        # Draw obstacle
        self.game_display.blit(self.obstacle_img, (self.obstacle_x, self.obstacle_y))
        
        # Draw car (with direction)
        if self.car_direction == -1:
            self.game_display.blit(self.car_left_img, (self.car_x, self.car_y))
        elif self.car_direction == 1:
            self.game_display.blit(self.car_right_img, (self.car_x, self.car_y))
        else:
            self.game_display.blit(self.car_img, (self.car_x, self.car_y))
        
        # Draw score
        font = pygame.font.SysFont(None, 25)
        score_text = font.render(f"Score: {self.score}", True, self.green_color)
        speed_text = font.render(f"Speed: {self.obstacle_speed:.1f}", True, self.green_color)
        frames_text = font.render(f"Frames: {self.frames_survived}", True, self.green_color)
        
        self.game_display.blit(score_text, (10, 10))
        self.game_display.blit(speed_text, (10, 35))
        self.game_display.blit(frames_text, (10, 60))
        
        pygame.display.update()
    
    def close(self):
        """Clean up pygame"""
        pygame.quit()

# Test the environment
if __name__ == "__main__":
    # Test the environment manually
    env = F1RaceEnvironment(render=True)
    
    running = True
    action = 0
    
    print("Manual Test Mode:")
    print("Use LEFT/RIGHT arrow keys to control the car")
    print("Press SPACE for no action")
    print("Press ESC or close window to exit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_RIGHT:
                    action = 2
                elif event.key == pygame.K_SPACE:
                    action = 0
        
        # Take action in environment
        state, reward, done, info = env.step(action)
        env.render()
        env.clock.tick(30)
        
        # Reset action to no-action after each step (prevents continuous movement)
        action = 0
        
        if done:
            print(f"Game Over! Score: {info['score']}, Frames: {info['frames']}")
            # Auto-reset for continuous testing
            env.reset()
    
    env.close()