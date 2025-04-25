# -*- coding: utf-8 -*-
"""
Keyboard UI module for displaying the virtual keyboard and
handling user interactions via eye tracking and blinks.
"""
import cv2
import numpy as np
import time
import logging
import math
import threading
import pygame
import os

# Check if running in a web environment (like Replit)
# If so, use the mock PyAutoGUI instead of the real one
if os.environ.get('REPL_ID') or not os.environ.get('DISPLAY'):
    from mock_pyautogui import press as pyautogui_press
    from mock_pyautogui import typewrite as pyautogui_typewrite
    from mock_pyautogui import get_typed_text, reset_typed_text
    import logging
    logging.info("Using mock PyAutoGUI for web environment")
else:
    # Use the real PyAutoGUI for desktop environments
    import pyautogui
    pyautogui_press = pyautogui.press
    pyautogui_typewrite = pyautogui.typewrite
    # Create dummy functions for compatibility
    def get_typed_text():
        return ""
    def reset_typed_text():
        pass

logger = logging.getLogger(__name__)

class KeyboardUI:
    """
    Virtual keyboard UI that can be controlled via eye tracking and blinks.
    """
    def __init__(self, eye_tracker, text_predictor, frame_width=1280, frame_height=720):
        self.eye_tracker = eye_tracker
        self.text_predictor = text_predictor
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Keyboard layout parameters
        self.keyboard_start_x = 10
        self.keyboard_start_y = 10
        self.key_width = 50
        self.key_height = 50
        self.key_gap = 5
        self.normal_keys_per_row = 5
        self.special_keys = ["<-", "SPC", "RST", "SRCH", "->"]  # Added -> for predictions
        
        # Scanning modes
        self.SCANNING_MODE_ROW = 0
        self.SCANNING_MODE_KEY = 1
        self.SCANNING_MODE_PREDICTION = 2
        
        # Timing parameters
        self.traversal_interval = 0.6
        self.post_blink_pause = 0.5
        
        # Colors
        self.key_color = (215, 215, 215)
        self.key_text_color = (0, 0, 0)
        self.highlight_color = (180, 255, 180)
        self.row_highlight_color = (200, 220, 255)
        self.special_key_color = (200, 200, 230)
        self.typed_text_color = (255, 255, 255)
        self.border_color = (100, 100, 100)
        self.prediction_color = (255, 230, 180)
        
        # State variables
        self.current_scan_mode = self.SCANNING_MODE_ROW
        self.current_row_scan_index = 0
        self.current_key_scan_index_in_row = 0
        self.current_prediction_index = 0
        self.selected_row_index = -1
        self.last_traversal_time = time.time()
        self.typed_text = ""
        self.is_paused = False
        self.predictions = []
        
        # Generate keyboard layout
        self.keyboard_rows = []
        self._generate_keyboard_layout()
        
        # Initialize audio feedback
        try:
            pygame.mixer.init()
            self.action_sound = None
            self.selection_sound = None
            self._initialize_sounds()
        except Exception as e:
            logger.warning(f"Could not initialize audio: {str(e)}")
            self.action_sound = None
            self.selection_sound = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        logger.info("Keyboard UI initialized")

    def _initialize_sounds(self):
        """Initialize audio feedback sounds."""
        try:
            # Action sound (key selection)
            self.action_sound = pygame.mixer.Sound("action.wav")
            logger.info("Action sound loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load action sound: {str(e)}")
            
        try:
            # Selection sound (row selection)
            self.selection_sound = pygame.mixer.Sound("selection.wav")
            logger.info("Selection sound loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load selection sound: {str(e)}")

    def _generate_keyboard_layout(self):
        """Generate the keyboard layout with letters and special keys."""
        logger.info("Generating keyboard layout...")
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        letter_idx = 0
        total_keys_in_layout = 0
        
        # Calculate number of rows needed for all letters
        num_letter_rows = math.ceil(len(letters) / self.normal_keys_per_row)
        
        for current_row_num in range(num_letter_rows):
            row_keys_chars = []
            current_row_layout = []
            
            # Add letters to this row
            for _ in range(self.normal_keys_per_row):
                if letter_idx < len(letters):
                    row_keys_chars.append(letters[letter_idx])
                    letter_idx += 1
                else:
                    row_keys_chars.append(None)
            
            # Add special keys to row
            row_keys_chars.extend(self.special_keys)
            
            # Generate key data for each character in row
            for col_idx, char in enumerate(row_keys_chars):
                if char is not None:
                    x = self.keyboard_start_x + col_idx * (self.key_width + self.key_gap)
                    y = self.keyboard_start_y + current_row_num * (self.key_height + self.key_gap)
                    
                    keyboard_data = {
                        "char": char,
                        "x": x,
                        "y": y,
                        "w": self.key_width,
                        "h": self.key_height,
                        "row": current_row_num,
                        "col": col_idx
                    }
                    
                    current_row_layout.append(keyboard_data)
                    total_keys_in_layout += 1
            
            if current_row_layout:
                self.keyboard_rows.append(current_row_layout)
        
        # Add a row for word predictions
        prediction_row = []
        for i in range(5):  # Space for 5 predictions
            x = self.keyboard_start_x + i * (2 * self.key_width + self.key_gap)
            y = self.keyboard_start_y + num_letter_rows * (self.key_height + self.key_gap) + 10
            
            prediction_data = {
                "char": f"PRED{i+1}",
                "x": x,
                "y": y,
                "w": 2 * self.key_width,
                "h": self.key_height,
                "row": num_letter_rows,
                "col": i
            }
            
            prediction_row.append(prediction_data)
            total_keys_in_layout += 1
            
        self.keyboard_rows.append(prediction_row)
        
        num_rows_actual = len(self.keyboard_rows)
        logger.info(f"Layout generated: {num_rows_actual} rows, {total_keys_in_layout} total keys")

    def play_sound(self, sound):
        """Play a sound effect."""
        if sound:
            try:
                sound.play()
            except pygame.error as e:
                logger.error(f"Error playing sound: {str(e)}")

    def update(self, frame):
        """
        Update the keyboard UI state and draw it on the given frame.
        Handles traversal through rows/keys and processes blink selections.
        """
        with self.lock:
            current_time = time.time()
            
            # Check if a blink was detected by the eye tracker
            blink_detected = False
            if hasattr(self.eye_tracker, 'blink_detector') and hasattr(self.eye_tracker, 'last_blink_time'):
                blink_time_diff = current_time - self.eye_tracker.last_blink_time
                if blink_time_diff < 0.1:  # Very recent blink
                    blink_detected = True
            
            # Handle pause after blink selection
            if self.is_paused and current_time - self.last_traversal_time >= self.post_blink_pause:
                self.is_paused = False
                self.last_traversal_time = current_time
            
            # Handle blink action for selection
            if blink_detected and not self.is_paused:
                self._handle_blink_selection()
                self.is_paused = True
                self.last_traversal_time = current_time
            
            # Handle traversal through rows/keys
            if not self.is_paused:
                if current_time - self.last_traversal_time >= self.traversal_interval:
                    self._handle_traversal()
                    self.last_traversal_time = current_time
            
            # Draw the keyboard UI
            self._draw_keyboard(frame)
            
            return frame

    def _handle_blink_selection(self):
        """Handle a blink detection for selection."""
        if self.current_scan_mode == self.SCANNING_MODE_ROW:
            # Select the current row
            if 0 <= self.current_row_scan_index < len(self.keyboard_rows):
                self.selected_row_index = self.current_row_scan_index
                logger.info(f"Row {self.selected_row_index} selected")
                self.current_scan_mode = self.SCANNING_MODE_KEY
                self.current_key_scan_index_in_row = 0
                self.play_sound(self.selection_sound)
        
        elif self.current_scan_mode == self.SCANNING_MODE_KEY:
            # Select the current key in the row
            if 0 <= self.selected_row_index < len(self.keyboard_rows):
                keys_in_row = self.keyboard_rows[self.selected_row_index]
                if 0 <= self.current_key_scan_index_in_row < len(keys_in_row):
                    key_info = keys_in_row[self.current_key_scan_index_in_row]
                    char = key_info["char"]
                    logger.info(f"Key '{char}' selected")
                    self.play_sound(self.action_sound)
                    
                    # Perform key action
                    action_taken = self._perform_key_action(char)
                    
                    # Reset to row scan if action was taken
                    if action_taken:
                        self._reset_to_row_scan()
        
        elif self.current_scan_mode == self.SCANNING_MODE_PREDICTION:
            # Select the current prediction
            if 0 <= self.current_prediction_index < len(self.predictions):
                selected_word = self.predictions[self.current_prediction_index]
                logger.info(f"Prediction '{selected_word}' selected")
                self.play_sound(self.action_sound)
                
                # Replace the last word with the prediction
                self._apply_prediction(selected_word)
                self._reset_to_row_scan()

    def _handle_traversal(self):
        """Handle traversal through rows/keys."""
        if self.current_scan_mode == self.SCANNING_MODE_ROW:
            # Traverse through rows
            self.current_row_scan_index = (self.current_row_scan_index + 1) % len(self.keyboard_rows)
        
        elif self.current_scan_mode == self.SCANNING_MODE_KEY:
            # Traverse through keys in the selected row
            keys_in_row = self.keyboard_rows[self.selected_row_index]
            self.current_key_scan_index_in_row = (self.current_key_scan_index_in_row + 1) % len(keys_in_row)
        
        elif self.current_scan_mode == self.SCANNING_MODE_PREDICTION:
            # Traverse through predictions
            if self.predictions:
                self.current_prediction_index = (self.current_prediction_index + 1) % len(self.predictions)

    def _perform_key_action(self, char):
        """Perform the action associated with the selected key."""
        action_taken = False
        
        # Handle special keys
        if char == "<-":
            # Backspace: remove last character
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
                pyautogui_press('backspace')
            action_taken = True
        
        elif char == "SPC":
            # Space: add space
            self.typed_text += " "
            pyautogui_press('space')
            
            # Get predictions after adding space
            self._update_predictions()
            action_taken = True
        
        elif char == "RST":
            # Reset: reset key index to start of current row
            self.current_key_scan_index_in_row = 0
            self.last_traversal_time = time.time()
            action_taken = False
        
        elif char == "SRCH":
            # Search: submit the text (e.g., press Enter)
            pyautogui_press('enter')
            action_taken = True
        
        elif char == "->":
            # Switch to prediction mode
            if self.predictions:
                self.current_scan_mode = self.SCANNING_MODE_PREDICTION
                self.current_prediction_index = 0
                logger.info("Switched to prediction mode")
                action_taken = False
            else:
                action_taken = True
        
        elif char.startswith("PRED"):
            # This is a prediction key placeholder, should not be selected directly
            action_taken = True
        
        else:
            # Regular character key
            self.typed_text += char
            pyautogui_press(char.lower())
            
            # Update predictions after adding character
            self._update_predictions()
            action_taken = True
        
        return action_taken

    def _update_predictions(self):
        """Update the word predictions based on current typed text."""
        if not self.text_predictor:
            return
            
        # Get predictions for the current text
        self.predictions = self.text_predictor.predict(self.typed_text)
        logger.debug(f"Updated predictions: {self.predictions}")

    def _apply_prediction(self, predicted_word):
        """Apply the selected prediction to the typed text."""
        # Find the last word in the typed text
        words = self.typed_text.split()
        if not words:
            # No words yet, just use the prediction
            self.typed_text = predicted_word
            self._type_with_pyautogui(predicted_word)
            return
        
        # Replace last word with prediction
        last_word = words[-1]
        backspaces_needed = len(last_word)
        
        # Delete the last word
        for _ in range(backspaces_needed):
            pyautogui_press('backspace')
        
        # Type the prediction
        self._type_with_pyautogui(predicted_word)
        
        # Update typed text
        self.typed_text = " ".join(words[:-1]) + (" " if words[:-1] else "") + predicted_word

    def _type_with_pyautogui(self, text):
        """Type the given text using pyautogui."""
        try:
            for char in text:
                if char == ' ':
                    pyautogui_press('space')
                else:
                    pyautogui_press(char.lower())
        except Exception as e:
            logger.error(f"Error typing with pyautogui: {str(e)}")

    def _reset_to_row_scan(self):
        """Reset to row scanning mode."""
        self.current_scan_mode = self.SCANNING_MODE_ROW
        self.current_row_scan_index = 0
        self.selected_row_index = -1
        self.current_key_scan_index_in_row = 0

    def _draw_keyboard(self, frame):
        """Draw the keyboard UI on the frame."""
        # Draw typed text area
        text_y = self.keyboard_start_y + len(self.keyboard_rows) * (self.key_height + self.key_gap) + 70
        cv2.rectangle(frame, 
                     (self.keyboard_start_x, text_y - 40), 
                     (self.frame_width - self.keyboard_start_x, text_y + 10),
                     (50, 50, 50),
                     -1)
        cv2.rectangle(frame, 
                     (self.keyboard_start_x, text_y - 40), 
                     (self.frame_width - self.keyboard_start_x, text_y + 10),
                     self.border_color,
                     2)
        
        # Draw typed text
        text_to_display = self.typed_text[-40:] if len(self.typed_text) > 40 else self.typed_text
        if text_to_display:
            cv2.putText(frame, text_to_display, 
                       (self.keyboard_start_x + 10, text_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.typed_text_color, 2)
        else:
            cv2.putText(frame, "Type with eye blinks...", 
                       (self.keyboard_start_x + 10, text_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        
        # Draw keyboard rows
        for row_idx, row in enumerate(self.keyboard_rows):
            # Highlight selected row or currently scanned row
            row_highlight = False
            if self.current_scan_mode == self.SCANNING_MODE_ROW and row_idx == self.current_row_scan_index:
                row_highlight = True
            elif self.selected_row_index == row_idx:
                row_highlight = True
            
            # Draw keys in this row
            for key_idx, key in enumerate(row):
                # Determine key color
                key_color = self.key_color
                
                if key["char"] in self.special_keys:
                    key_color = self.special_key_color
                elif key["char"].startswith("PRED"):
                    key_color = self.prediction_color
                
                # Highlight current key if being scanned
                if (self.current_scan_mode == self.SCANNING_MODE_KEY and 
                    row_idx == self.selected_row_index and 
                    key_idx == self.current_key_scan_index_in_row):
                    key_color = self.highlight_color
                elif row_highlight:
                    key_color = self.row_highlight_color
                
                # Draw key rectangle
                cv2.rectangle(frame, (key["x"], key["y"]), 
                             (key["x"] + key["w"], key["y"] + key["h"]),
                             key_color, -1)
                cv2.rectangle(frame, (key["x"], key["y"]), 
                             (key["x"] + key["w"], key["y"] + key["h"]),
                             self.border_color, 1)
                
                # Draw key text
                display_char = key["char"]
                if key["char"].startswith("PRED") and row_idx == len(self.keyboard_rows) - 1:
                    # Show actual prediction text for prediction keys
                    pred_idx = int(key["char"][4:]) - 1
                    if 0 <= pred_idx < len(self.predictions):
                        display_char = self.predictions[pred_idx]
                        
                        # Highlight current prediction if in prediction mode
                        if (self.current_scan_mode == self.SCANNING_MODE_PREDICTION and 
                            pred_idx == self.current_prediction_index):
                            cv2.rectangle(frame, (key["x"], key["y"]), 
                                         (key["x"] + key["w"], key["y"] + key["h"]),
                                         self.highlight_color, -1)
                            cv2.rectangle(frame, (key["x"], key["y"]), 
                                         (key["x"] + key["w"], key["y"] + key["h"]),
                                         self.border_color, 1)
                
                # Draw text at center of key
                text_size = cv2.getTextSize(display_char, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = key["x"] + (key["w"] - text_size[0]) // 2
                text_y = key["y"] + (key["h"] + text_size[1]) // 2
                
                cv2.putText(frame, display_char, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.key_text_color, 2)
        
        # Draw instructions
        instruction_y = text_y + 60
        if self.current_scan_mode == self.SCANNING_MODE_ROW:
            instruction = "Blink to select ROW"
        elif self.current_scan_mode == self.SCANNING_MODE_KEY:
            instruction = "Blink to select KEY"
        else:
            instruction = "Blink to select PREDICTION"
            
        cv2.putText(frame, instruction, 
                   (self.keyboard_start_x + 10, instruction_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
