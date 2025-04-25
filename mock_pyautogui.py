# -*- coding: utf-8 -*-
"""
Mock PyAutoGUI module for web environments.
Provides mock implementations of PyAutoGUI functions used in the application
to allow the web interface to function without X11 dependency.
"""
import logging

logger = logging.getLogger(__name__)

class MockPyAutoGUI:
    """Mock implementation of PyAutoGUI for headless environments."""
    
    def __init__(self):
        self.last_action = None
        self.typed_text = ""
        logger.info("Initialized MockPyAutoGUI for web environment")
    
    def press(self, key):
        """Mock implementation of pyautogui.press()."""
        logger.info(f"Mock PyAutoGUI: Pressed key '{key}'")
        self.last_action = f"press_{key}"
        
        # Track typed text for display/debugging
        if key == 'backspace':
            if self.typed_text:
                self.typed_text = self.typed_text[:-1]
        elif key == 'space':
            self.typed_text += " "
        elif key == 'enter':
            self.typed_text += "\n"
        elif len(key) == 1:  # Single character key
            self.typed_text += key
    
    def typewrite(self, text):
        """Mock implementation of pyautogui.typewrite()."""
        logger.info(f"Mock PyAutoGUI: Typed '{text}'")
        self.last_action = f"type_{text}"
        self.typed_text += text
    
    def hotkey(self, *args):
        """Mock implementation of pyautogui.hotkey()."""
        key_combo = '+'.join(args)
        logger.info(f"Mock PyAutoGUI: Hotkey '{key_combo}'")
        self.last_action = f"hotkey_{key_combo}"
    
    def get_typed_text(self):
        """Get the text that would have been typed."""
        return self.typed_text
    
    def reset_typed_text(self):
        """Reset the typed text buffer."""
        self.typed_text = ""

# Create global instance
mock_pyautogui = MockPyAutoGUI()

# Expose functions like the real PyAutoGUI
def press(key):
    """Mock implementation of pyautogui.press()."""
    return mock_pyautogui.press(key)

def typewrite(text):
    """Mock implementation of pyautogui.typewrite()."""
    return mock_pyautogui.typewrite(text)

def hotkey(*args):
    """Mock implementation of pyautogui.hotkey()."""
    return mock_pyautogui.hotkey(*args)

def get_typed_text():
    """Get the text that would have been typed."""
    return mock_pyautogui.get_typed_text()

def reset_typed_text():
    """Reset the typed text buffer."""
    return mock_pyautogui.reset_typed_text()