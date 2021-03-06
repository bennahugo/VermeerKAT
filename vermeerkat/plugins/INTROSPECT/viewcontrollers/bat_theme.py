import npyscreen
import curses


class bat_theme(npyscreen.ThemeManager):
    default_colors = {
        'DEFAULT': 'WHITE_BLACK',
        'FORMDEFAULT': 'WHITE_BLACK',
        'MBOXDEFAULT': 'RED_WHITE',
        'NO_EDIT': 'WHITE_BLACK',
        'STANDOUT': 'RED_BLACK',
        'CURSOR': 'YELLOW_WHITE',
        'CURSOR_INVERSE': 'BLACK_YELLOW',
        'LABEL': 'WHITE_BLACK',
        'LABELBOLD': 'WHITE_BLACK',
        'CONTROL': 'YELLOW_BLACK',
        'IMPORTANT': 'GREEN_BLACK',
        'SAFE': 'GREEN_BLACK',
        'WARNING': 'YELLOW_BLACK',
        'DANGER': 'RED_BLUE',
        'CRITICAL': 'RED_YELLOW',
        'GOOD': 'GREEN_BLACK',
        'GOODHL': 'GREEN_BLACK',
        'VERYGOOD': 'BLACK_GREEN',
        'CAUTION': 'YELLOW_BLUE',
        'CAUTIONHL': 'BLACK_YELLOW',
    }

    _colors_to_define = (
        # DO NOT DEFINE THE WHITE_BLACK COLOR - THINGS BREAK
        #('WHITE_BLACK',      DO_NOT_DO_THIS,      DO_NOT_DO_THIS),
        ('BLACK_BLUE',      curses.COLOR_BLACK,      curses.COLOR_BLUE),
        ('BLACK_WHITE',      curses.COLOR_WHITE,      curses.COLOR_BLACK),
        #('BLACK_ON_DEFAULT', curses.COLOR_BLACK,      -1),
        #('WHITE_ON_DEFAULT', curses.COLOR_WHITE,      -1),
        ('BLUE_BLACK',       curses.COLOR_BLUE,       curses.COLOR_BLACK),
        ('WHITE_BLUE',      curses.COLOR_WHITE,      curses.COLOR_BLUE),
        ('CYAN_BLUE',       curses.COLOR_CYAN,       curses.COLOR_BLUE),
        ('CYAN_BLACK',       curses.COLOR_CYAN,       curses.COLOR_BLACK),
        ('GREEN_BLUE',      curses.COLOR_GREEN,      curses.COLOR_BLUE),
        ('GREEN_BLACK',      curses.COLOR_GREEN,      curses.COLOR_BLACK),
        ('MAGENTA_BLUE',    curses.COLOR_MAGENTA,    curses.COLOR_BLUE),
        ('MAGENTA_BLACK',    curses.COLOR_MAGENTA,    curses.COLOR_BLACK),
        ('RED_BLUE',        curses.COLOR_RED,        curses.COLOR_BLUE),
        ('RED_BLACK',        curses.COLOR_RED,        curses.COLOR_BLACK),
        ('YELLOW_BLUE',     curses.COLOR_YELLOW,     curses.COLOR_BLUE),
        ('YELLOW_BLACK',     curses.COLOR_YELLOW,     curses.COLOR_BLACK),
        ('BLACK_YELLOW',     curses.COLOR_BLACK,     curses.COLOR_YELLOW),
        ('BLUE_RED',        curses.COLOR_BLUE,      curses.COLOR_RED),
        ('BLACK_RED',        curses.COLOR_BLACK,      curses.COLOR_RED),
        ('BLUE_GREEN',      curses.COLOR_BLUE,      curses.COLOR_GREEN),
        ('BLACK_GREEN',      curses.COLOR_BLACK,      curses.COLOR_GREEN),
        ('BLUE_YELLOW',     curses.COLOR_BLUE,      curses.COLOR_YELLOW),
        ('BLACK_YELLOW',     curses.COLOR_BLACK,      curses.COLOR_YELLOW),
        ('BLUE_CYAN',       curses.COLOR_BLUE,       curses.COLOR_CYAN),
        ('BLUE_WHITE',       curses.COLOR_BLUE,       curses.COLOR_WHITE),
        ('BLACK_WHITE',       curses.COLOR_BLACK,       curses.COLOR_WHITE),
        ('CYAN_WHITE',       curses.COLOR_CYAN,       curses.COLOR_WHITE),
        ('GREEN_WHITE',      curses.COLOR_GREEN,      curses.COLOR_WHITE),
        ('MAGENTA_WHITE',    curses.COLOR_MAGENTA,    curses.COLOR_WHITE),
        ('RED_WHITE',        curses.COLOR_RED,        curses.COLOR_WHITE),
        ('YELLOW_WHITE',     curses.COLOR_YELLOW,     curses.COLOR_WHITE),
        ('WHITE_YELLOW',     curses.COLOR_WHITE,     curses.COLOR_YELLOW),
    )


