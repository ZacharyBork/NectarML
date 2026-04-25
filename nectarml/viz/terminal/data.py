from enum import Enum

class Codes(Enum):
    RED    = '\u001b[38;2;255;0;0m'
    YELLOW = '\u001b[38;2;255;255;0m'
    GREEN  = '\u001b[38;2;0;255;0m'
    TEAL   = '\u001b[38;2;0;255;255m'
    BLUE   = '\u001b[38;2;0;0;255m'
    PURPLE = '\u001b[38;2;255;0;255m'
    
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'

    
