"""
🏎️ F1 Race Road Game AI Environment 🤖
=========================================

Welcome to an exciting journey into Reinforcement Learning! 🎮✨

This module creates a custom OpenAI Gym-style environment where an AI agent learns to drive
a Formula 1 car and dodge obstacles. Think of it as teaching a computer to play a racing game,
just like how you might teach a friend - through trial and error, rewards, and practice!

🎯 WHAT IS REINFORCEMENT LEARNING?
----------------------------------
Imagine teaching someone to ride a bicycle:
- They try something (ACTION) → like turning the handlebars
- They see what happens (STATE) → like starting to fall left
- They get feedback (REWARD) → like "good balance!" or "oops, you're falling!"
- They learn and try again, getting better each time

That's exactly what our AI does with this racing game! 🚲➡️🏎️

🧠 THE AI LEARNING PROCESS:
1. 👀 OBSERVE: AI looks at the game screen (car position, obstacle location, speed)
2. 🤔 DECIDE: AI chooses an action (move left, right, or stay)
3. 🎬 ACT: The game updates based on the AI's choice
4. 📊 LEARN: AI gets a reward/penalty and updates its "brain" (neural network)
5. 🔄 REPEAT: Do this thousands of times until the AI becomes a pro driver!

🎮 WHY THIS GAME?
-----------------
- Simple rules = easier to learn AI concepts
- Visual feedback = you can watch the AI improve in real-time
- Gradual difficulty = obstacles get faster as score increases
- Clear success metric = how long can the AI survive?

Perfect for understanding AI without getting lost in complexity! 🎓

👥 AUDIENCE NOTES:
- 🔬 Data Scientists: Notice the state normalization and reward shaping
- 👩‍🏫 Educators: Great example of trial-and-error learning
- 👶 Young Coders: It's like teaching a robot to play your favorite mobile game!
- 🤓 AI Curious: See how machines learn through interaction, not just memorization

Author: Pat Snyder 💻
Created for: Learning Labs Portfolio 🌟
"""

import pygame
import random
import numpy as np
import os
import math

class F1RaceEnvironment:
    """
    🏁 F1 Race Game Environment for AI Training 🤖
    
    This class creates a simple racing game where:
    - A blue car (AI player) tries to avoid red obstacles
    - The car can move left, right, or stay in place
    - Speed increases as the AI gets better
    - The AI gets points for surviving and avoiding crashes
    
    Perfect for learning how AI agents interact with environments! 🎯
    """
    
    def __init__(self, render=True):
        """
        🎮 Initialize the F1 Race Game Environment
        
        Args:
            render (bool): Whether to show the game window (True) or run headless (False)
                          Set to False when training AI for faster learning!
        """
        pygame.init()
        
        # 🎛️ GAME CONFIGURATION - Easy to modify! 
        # ========================================
        self.SCREEN_WIDTH = 400          # 📏 Game window width
        self.SCREEN_HEIGHT = 600         # 📏 Game window height  
        self.CAR_SPEED = 8              # 🚗 How fast car moves left/right
        self.INITIAL_OBSTACLE_SPEED = 5  # 🚧 Starting obstacle speed
        self.SPEED_INCREASE = 0.5        # 📈 Speed increase per obstacle passed
        self.MAX_SPEED = 20             # 🏎️ Maximum obstacle speed
        
        # 🎨 COLOR PALETTE - RGB values (Red, Green, Blue)
        # ===============================================
        self.BLACK = (0, 0, 0)          # Background
        self.WHITE = (255, 255, 255)    # Text/UI
        self.RED = (255, 0, 0)          # Obstacles
        self.GREEN = (0, 255, 0)        # Score display
        self.BLUE = (0, 0, 255)         # Player car
        self.GRAY = (64, 64, 64)        # Road texture
        
        # 🏆 REWARD SYSTEM - How AI gets feedback
        # ======================================
        self.CRASH_PENALTY = -100       # 💥 Big penalty for crashing
        self.SURVIVAL_REWARD = 0.1      # ✨ Small reward for each frame survived
        self.DODGE_BONUS = 10           # 🎯 Bonus for successfully dodging obstacle
        self.MOVEMENT_COST = -0.01      # 🔋 Small cost for unnecessary movement
        
        # 🎬 SETUP DISPLAY
        # ================
        self.render_game = render
        if self.render_game:
            # Create game window that player can see
            self.game_display = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption('🏎️ F1 Race AI Training')
        else:
            # Create invisible surface for faster AI training
            self.game_display = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            
        self.clock = pygame.time.Clock()
        
        # 🖼️ LOAD GAME IMAGES
        # ===================
        self.load_game_images()
        
        # 📐 GET IMAGE SIZES
        # ==================
        car_rect = self.car_img.get_rect()
        self.car_width = car_rect.width
        self.car_height = car_rect.height
        
        obstacle_rect = self.obstacle_img.get_rect()
        self.obstacle_width = obstacle_rect.width  
        self.obstacle_height = obstacle_rect.height
        
        # 🎯 AI CONFIGURATION
        # ===================
        self.action_space_size = 3      # 0=stay, 1=left, 2=right
        self.state_space_size = 5       # 5 numbers describing game state
        
        # 🎮 START THE GAME!
        # ==================
        self.reset()
        
    def load_game_images(self):
        """
        🖼️ Load all game images (car, obstacles, background)
        If images don't exist, creates simple colored rectangles instead
        """
        try:
            # Try to load actual image files
            self.car_img = pygame.image.load('images/car.png')
            self.car_left_img = pygame.image.load('images/car_left.png') 
            self.car_right_img = pygame.image.load('images/car_right.png')
            self.obstacle_img = pygame.image.load('images/obstacle.png')
            self.background_img = pygame.image.load('images/background.png')
            self.texture_img = pygame.image.load('images/texture.png')
            
        except pygame.error as error:
            print(f"⚠️  Could not load images: {error}")
            print("📦 Creating simple colored rectangles instead...")
            
            # Create simple colored shapes if images fail
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
        
    def reset(self):
        """
        🔄 Reset game to starting position
        
        Returns:
            numpy.array: Starting game state for the AI
        """
        # 🚗 CAR STARTING POSITION
        # ========================
        self.car_x = self.SCREEN_WIDTH * 0.4  # Start towards left-center
        self.car_y = self.SCREEN_HEIGHT * 0.8  # Near bottom of screen
        self.car_direction = 0  # 0=straight, -1=left, 1=right (for animation)
        
        # 🚧 OBSTACLE STARTING POSITION  
        # =============================
        safe_left = 8  # Don't spawn too close to edge
        safe_right = self.SCREEN_WIDTH - self.obstacle_width - 8
        self.obstacle_x = random.randrange(safe_left, safe_right)
        self.obstacle_y = -600  # Start way above screen
        self.obstacle_speed = self.INITIAL_OBSTACLE_SPEED
        
        # 📊 GAME STATISTICS
        # ==================
        self.score = 0
        self.game_over = False
        self.frames_survived = 0
        self.texture_y = 0  # For moving road effect
        
        return self.get_current_state()
    
    def get_current_state(self):
        """
        📊 Get current game state as numbers the AI can understand
        
        The AI sees the game as 5 numbers between 0 and 1:
        1. Where is my car? (0=far left, 1=far right)
        2. Where is the obstacle? (0=far left, 1=far right)  
        3. How close is the obstacle? (0=very far, 1=very close)
        4. How fast are obstacles moving? (0=slow, 1=fast)
        5. What's my distance from the obstacle? (0=collision, 1=safe)
        
        Returns:
            numpy.array: 5 normalized values representing game state
        """
        # 📍 NORMALIZE POSITIONS (convert to 0-1 scale)
        # ============================================
        car_position = self.car_x / self.SCREEN_WIDTH
        obstacle_position = self.obstacle_x / self.SCREEN_WIDTH
        
        # 📏 NORMALIZE OBSTACLE DISTANCE  
        # ==============================
        # Obstacle can be from -600 (way above) to SCREEN_HEIGHT (way below)
        total_possible_distance = self.SCREEN_HEIGHT + 600
        obstacle_closeness = (self.obstacle_y + 600) / total_possible_distance
        
        # 🏎️ NORMALIZE SPEED
        # ==================
        speed_ratio = min(self.obstacle_speed / self.MAX_SPEED, 1.0)
        
        # 📐 CALCULATE DISTANCE TO OBSTACLE
        # =================================
        horizontal_distance = abs(self.car_x - self.obstacle_x)
        vertical_distance = abs(self.car_y - self.obstacle_y)
        total_distance = math.sqrt(horizontal_distance**2 + vertical_distance**2)
        
        # Maximum possible distance is corner to corner
        max_distance = math.sqrt(self.SCREEN_WIDTH**2 + self.SCREEN_HEIGHT**2)
        distance_ratio = total_distance / max_distance
        
        # 📦 PACKAGE STATE FOR AI
        # =======================
        state = np.array([
            car_position,          # My car's position
            obstacle_position,     # Obstacle's position  
            obstacle_closeness,    # How close obstacle is
            speed_ratio,          # How fast game is going
            distance_ratio        # Overall distance to obstacle
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """
        🎬 Execute one game step with AI's chosen action
        
        This is the main game loop function where:
        1. AI chooses an action (0, 1, or 2)
        2. Game updates based on that action  
        3. AI gets feedback (reward/penalty)
        4. Game provides new state information
        
        Args:
            action (int): 0=stay still, 1=move left, 2=move right
            
        Returns:
            tuple: (new_state, reward, game_over, info_dict)
        """
        # ⏹️ CHECK IF GAME ALREADY OVER
        # =============================
        if self.game_over:
            return self.get_current_state(), 0, True, {}
        
        # 🎮 HANDLE PYGAME EVENTS (for window closing)
        # ==========================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
        
        # 🚗 MOVE THE CAR BASED ON AI'S ACTION
        # ====================================
        self.move_car(action)
        
        # 🎯 UPDATE GAME WORLD
        # ====================
        self.update_game_objects()
        
        # 🏆 CALCULATE AI'S REWARD
        # ========================
        reward = self.calculate_reward()
        
        # 📊 PREPARE RESULTS FOR AI
        # =========================
        next_state = self.get_current_state()
        is_done = self.game_over
        info = {
            "score": self.score,
            "frames": self.frames_survived,
            "speed": self.obstacle_speed
        }
        
        self.frames_survived += 1
        
        return next_state, reward, is_done, info
    
    def move_car(self, action):
        """
        🚗 Move the car based on AI's action choice
        
        Args:
            action (int): 0=stay, 1=go left, 2=go right
        """
        if action == 1:  # Move left
            self.car_x = self.car_x - self.CAR_SPEED
            self.car_direction = -1
            
            # Don't let car go off the left edge
            if self.car_x < 0:
                self.car_x = 0
                
        elif action == 2:  # Move right  
            self.car_x = self.car_x + self.CAR_SPEED
            self.car_direction = 1
            
            # Don't let car go off the right edge
            car_right_edge = self.SCREEN_WIDTH - self.car_width
            if self.car_x > car_right_edge:
                self.car_x = car_right_edge
                
        else:  # Stay still (action == 0)
            self.car_direction = 0
    
    def update_game_objects(self):
        """
        🎯 Update all moving parts of the game each frame
        - Move obstacle down the screen
        - Check if obstacle was dodged (score!)
        - Check for collisions
        - Increase difficulty over time
        """
        # 🚧 MOVE OBSTACLE DOWN
        # ====================
        self.obstacle_y = self.obstacle_y + self.obstacle_speed
        
        # 🛣️ MOVE ROAD TEXTURE FOR VISUAL EFFECT
        # ======================================
        self.texture_y = self.texture_y + self.obstacle_speed
        if self.texture_y >= 400:
            self.texture_y = 0
        
        # 🎯 CHECK IF OBSTACLE WAS SUCCESSFULLY DODGED
        # ===========================================
        if self.obstacle_y > self.SCREEN_HEIGHT:
            # Success! AI dodged the obstacle
            self.score = self.score + 1
            
            # Create new obstacle at top
            safe_left = 8
            safe_right = self.SCREEN_WIDTH - self.obstacle_width - 8  
            self.obstacle_x = random.randrange(safe_left, safe_right)
            self.obstacle_y = -self.obstacle_height
            
            # Make game slightly harder
            new_speed = self.obstacle_speed + self.SPEED_INCREASE
            self.obstacle_speed = min(new_speed, self.MAX_SPEED)
        
        # 💥 CHECK FOR COLLISION
        # =====================
        if self.check_collision():
            self.game_over = True
        
        # 🚫 CHECK IF CAR WENT OFF SCREEN (also game over)
        # ===============================================
        if self.car_x < 0 or self.car_x > (self.SCREEN_WIDTH - self.car_width):
            self.game_over = True
    
    def check_collision(self):
        """
        💥 Check if the car hit an obstacle
        
        Returns:
            bool: True if collision happened, False if safe
        """
        # Create rectangles for collision detection
        car_rectangle = pygame.Rect(self.car_x, self.car_y, self.car_width, self.car_height)
        obstacle_rectangle = pygame.Rect(self.obstacle_x, self.obstacle_y, self.obstacle_width, self.obstacle_height)
        
        # Check if rectangles overlap
        collision_happened = car_rectangle.colliderect(obstacle_rectangle)
        return collision_happened
    
    def calculate_reward(self):
        """
        🏆 Calculate reward/penalty for AI's current performance
        
        This is how the AI learns what's good and bad:
        - Big penalty for crashing (teaches "don't crash!")  
        - Small reward for surviving (teaches "stay alive!")
        - Big bonus for dodging obstacles (teaches "avoid obstacles!")
        - Small penalty for moving unnecessarily (teaches "be efficient!")
        
        Returns:
            float: The reward value (positive=good, negative=bad)
        """
        # 💥 CRASHED = BIG PENALTY
        # ========================
        if self.game_over:
            return self.CRASH_PENALTY
        
        # ✨ BASE REWARD FOR SURVIVING
        # ===========================
        reward = self.SURVIVAL_REWARD
        
        # 🎯 BONUS FOR SUCCESSFULLY DODGING OBSTACLE  
        # ==========================================
        # Check if obstacle just passed the car (successful dodge)
        obstacle_just_passed = (self.obstacle_y > self.car_y and 
                              self.obstacle_y <= self.car_y + self.obstacle_speed)
        
        if obstacle_just_passed:
            reward = reward + self.DODGE_BONUS
        
        # 🔋 SMALL PENALTY FOR UNNECESSARY MOVEMENT
        # ========================================
        if self.car_direction != 0:  # Car is moving left or right
            reward = reward + self.MOVEMENT_COST
            
        return reward
    
    def render(self):
        """
        🎨 Draw the game on screen (only if render_game=True)
        
        This creates the visual representation that humans can watch!
        """
        # Skip rendering if running headless (for faster AI training)
        if not self.render_game:
            return
            
        # 🖼️ DRAW BACKGROUND
        # ==================
        self.game_display.blit(self.background_img, (0, 0))
        
        # 🛣️ DRAW MOVING ROAD TEXTURE
        # ===========================
        self.game_display.blit(self.texture_img, (0, self.texture_y - 400))
        self.game_display.blit(self.texture_img, (0, self.texture_y))
        self.game_display.blit(self.texture_img, (0, self.texture_y + 400))
        
        # 🚧 DRAW OBSTACLE
        # ================
        self.game_display.blit(self.obstacle_img, (self.obstacle_x, self.obstacle_y))
        
        # 🚗 DRAW CAR (with direction animation)
        # =====================================
        if self.car_direction == -1:        # Moving left
            car_image = self.car_left_img
        elif self.car_direction == 1:       # Moving right
            car_image = self.car_right_img  
        else:                              # Going straight
            car_image = self.car_img
            
        self.game_display.blit(car_image, (self.car_x, self.car_y))
        
        # 📊 DRAW GAME STATISTICS
        # =======================
        font = pygame.font.SysFont(None, 25)
        
        # Create text surfaces
        score_display = font.render(f"Score: {self.score}", True, self.GREEN)
        speed_display = font.render(f"Speed: {self.obstacle_speed:.1f}", True, self.GREEN)
        frames_display = font.render(f"Frames: {self.frames_survived}", True, self.GREEN)
        
        # Draw text on screen
        self.game_display.blit(score_display, (10, 10))
        self.game_display.blit(speed_display, (10, 35))
        self.game_display.blit(frames_display, (10, 60))
        
        # 🖥️ UPDATE DISPLAY
        # =================
        pygame.display.update()
    
    def close(self):
        """🚪 Clean up and close the game"""
        pygame.quit()

# 🧪 TEST THE ENVIRONMENT
# =======================
if __name__ == "__main__":
    """
    This section runs when you execute this file directly
    It creates a manual testing mode where you can play the game yourself!
    """
    # Create the game environment
    env = F1RaceEnvironment(render=True)
    
    # Game control variables
    running = True
    action = 0
    
    print("🎮 MANUAL TEST MODE ACTIVATED!")
    print("================================")
    print("🏎️  Use LEFT/RIGHT arrow keys to control the car")
    print("⏸️  Press SPACE for no action")
    print("🚪 Press ESC or close window to exit")
    print("🎯 Try to dodge the red obstacles!")
    print()
    
    # Main game loop
    while running:
        # 🎮 HANDLE PLAYER INPUT
        # ======================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    action = 1  # Move left
                elif event.key == pygame.K_RIGHT:
                    action = 2  # Move right
                elif event.key == pygame.K_SPACE:
                    action = 0  # Stay still
        
        # 🎬 TAKE ACTION IN GAME
        # =====================
        state, reward, done, info = env.step(action)
        env.render()
        env.clock.tick(30)  # 30 FPS
        
        # Reset action (prevents continuous movement)
        action = 0
        
        # 🎯 HANDLE GAME OVER
        # ===================
        if done:
            final_score = info['score']
            total_frames = info['frames']
            print(f"💥 Game Over! Final Score: {final_score}, Frames Survived: {total_frames}")
            print("🔄 Auto-restarting for continuous testing...")
            env.reset()
    
    # 🚪 CLEANUP
    # ==========