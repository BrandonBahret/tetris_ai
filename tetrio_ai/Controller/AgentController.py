import time

import Controller.DirectKeyboard as keyboard
from Controller.MoveDescription import MoveDescription

from Types.Singleton import Singleton

class TimedCondition:
    def __init__(self):
        self._last_time_t = -1
        self._condition_met = False

    def update(self, conditional):
        if not self._condition_met and conditional:
            self._last_time_t = time.time()
            self._condition_met = True

        elif not conditional:
            self._last_time_t = -1
            self._condition_met = False

    @property
    def elapsed_time(self):
        if self._last_time_t == -1:
            return -1
        
        return time.time() - self._last_time_t

    def has_been_true(self, time_in_seconds):
        if self.elapsed_time >= time_in_seconds:
            return self._condition_met

        return False

class ConditionMonitor(metaclass=Singleton):
    def __init__(self):
        self.conditions_dict = dict()

    @staticmethod
    def update(name, conditional):
        monitor = ConditionMonitor()
        if not name in monitor.conditions_dict:
            monitor.conditions_dict[name] = TimedCondition()

        monitor.conditions_dict[name].update(conditional)

def condition(name):
    monitor = ConditionMonitor()
    return monitor.conditions_dict[name]


class Action:
    def __init__(self, key_code):
        self._key_code = key_code
        self._last_key_time_t = time.time()

    def time_elapsed(self):
        return time.time() - self._last_key_time_t

    def press_key(self, cooldown=0, duration=0):
        if self.time_elapsed() > cooldown:
            self._last_key_time_t = time.time()
            keyboard.PressKey(self._key_code)
            time.sleep(duration)
            keyboard.ReleaseKey(self._key_code)

class AgentController(metaclass=Singleton):
    '''Agent Controller translates actions taken by the Game Agent.'''
    
    def __init__(self):
        self.rotate_right = Action(keyboard.key_l)
        self.rotate_left = Action(keyboard.key_j)
        self.rotate_180 = Action(keyboard.key_k)
        self.move_right = Action(keyboard.key_d)
        self.move_left = Action(keyboard.key_a)
        self.hard_drop = Action(keyboard.key_slash)
        self.soft_drop = Action(keyboard.key_comma)
        self.swap_hold_piece = Action(keyboard.key_s)
        self.reset_game = Action(keyboard.key_f1)

    def reset(self):
        self.reset_game.press_key(cooldown=15.00, duration=0.5)

    def translate(self, active_piece, target_move:MoveDescription):
        target_move.check_progress(active_piece)
        if target_move.move_complete:
            return None

        if target_move.destination.is_held_piece:
            self.swap_hold_piece.press_key(cooldown=0.01)
            target_move.increment_progress_index()
        else:
            self.move(active_piece, target_move)
            self.rotate(active_piece, target_move)

    def rotate(self, active_piece, target_move):
        if target_move.move_complete:
            return None

        current_rotation = active_piece.orientation
        target_rotation = target_move.destination.orientation
        if current_rotation == target_rotation:
            return None

        rot180 = (current_rotation + 2) % 4
        current_delta = abs(target_rotation - current_rotation)
        rot180_delta = abs(target_rotation - rot180)
    
        if rot180_delta < current_delta:
            self.rotate_180.press_key(cooldown=0.13)

        elif target_rotation > current_rotation:
            self.rotate_left.press_key(cooldown=0.13)

        elif target_rotation < current_rotation:
            self.rotate_right.press_key(cooldown=0.13)

    def move(self, active_piece, target_move):
        if target_move.move_complete:
            return None

        current_position = active_piece.position
        target_position = target_move.destination.position            
        target_x, _ = target_position
        current_x, _ = current_position

        ConditionMonitor.update("should_drop", current_x == target_x)
        if condition("should_drop").has_been_true(0.10):
            if target_move.is_last_step:
                self.hard_drop.press_key(cooldown=0.35, duration=0.01)
                ConditionMonitor.update("should_drop", False)

            elif target_move.vertical_distance_from_destination(active_piece) >= 3:
                self.soft_drop.press_key(cooldown=0.08, duration=0.01)
        
        ConditionMonitor.update("may_move", current_x != target_x)
        if condition("may_move").has_been_true(0.15):
            if target_x > current_x:
                self.move_right.press_key(cooldown=0.08)

            elif target_x < current_x:
                self.move_left.press_key(cooldown=0.08)
