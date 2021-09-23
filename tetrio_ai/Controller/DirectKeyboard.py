# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes
import time

import WindowCamera

TARGET_PROCESS_NAME = "TETR.IO.exe"

def appropriate_window_focused():
    try:
        window = WindowCamera.find_active_window()
        return window.process_name == TARGET_PROCESS_NAME
    except ValueError as e:
        print(e)
        return False

SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    if appropriate_window_focused():
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

key_escape     = 0x01
key_1          = 0x02
key_2          = 0x03
key_3          = 0x04
key_4          = 0x05
key_5          = 0x06
key_6          = 0x07
key_7          = 0x08
key_8          = 0x09
key_9          = 0x0a
key_0          = 0x0b
key_minus      = 0x0c    #/* - on main keyboard */
key_equals     = 0x0d
key_back       = 0x0e    #/* backspace */
key_tab        = 0x0f
key_q          = 0x10
key_w          = 0x11
key_e          = 0x12
key_r          = 0x13
key_t          = 0x14
key_y          = 0x15
key_u          = 0x16
key_i          = 0x17
key_o          = 0x18
key_p          = 0x19
key_lbracket   = 0x1a
key_rbracket   = 0x1b
key_return     = 0x1c    #/* enter on main keyboard */
key_lcontrol   = 0x1d
key_a          = 0x1e
key_s          = 0x1f
key_d          = 0x20
key_f          = 0x21
key_g          = 0x22
key_h          = 0x23
key_j          = 0x24
key_k          = 0x25
key_l          = 0x26
key_semicolon  = 0x27
key_apostrophe = 0x28
key_grave      = 0x29    #/* accent grave */
key_lshift     = 0x2a
key_backslash  = 0x2b
key_z          = 0x2c
key_x          = 0x2d
key_c          = 0x2e
key_v          = 0x2f
key_b          = 0x30
key_n          = 0x31
key_m          = 0x32
key_comma      = 0x33
key_period     = 0x34    #/* . on main keyboard */
key_slash      = 0x35    #/* / on main keyboard */
key_rshift     = 0x36
key_multiply   = 0x37    #/* * on numeric keypad */
key_lmenu      = 0x38    #/* left alt */
key_space      = 0x39
key_capital    = 0x3a
key_f1         = 0x3b
key_f2         = 0x3c
key_f3         = 0x3d
key_f4         = 0x3e
key_f5         = 0x3f
key_f6         = 0x40
key_f7         = 0x41
key_f8         = 0x42
key_f9         = 0x43
key_f10        = 0x44
key_numlock    = 0x45
key_scroll     = 0x46    #/* scroll lock */
key_numpad7    = 0x47
key_numpad8    = 0x48
key_numpad9    = 0x49
key_subtract   = 0x4a    #/* - on numeric keypad */
key_numpad4    = 0x4b
key_numpad5    = 0x4c
key_numpad6    = 0x4d
key_add        = 0x4e    #/* + on numeric keypad */
key_numpad1    = 0x4f
key_numpad2    = 0x50
key_numpad3    = 0x51
key_numpad0    = 0x52
key_decimal    = 0x53    #/* . on numeric keypad */
key_f11        = 0x57
key_f12        = 0x58
