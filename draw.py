from turtle import *

s = 50


def cntstrs(strings):
    return len([item for item in strings if type(item) is str])


def draw_tree(tree, pos, head=0):
    c = cntstrs(tree)
    while len(tree):
        goto(pos)
        item = tree.pop(0)
        if head:
            write(item, 1)
            if len(tree) == 0:
                break
            draw_tree(tree.pop(0), pos)
        else:
            if type(item) is str:
                newpos = (pos[0] + s * c / 4 - s * cntstrs(tree), pos[1] - s)
                down()
                goto((newpos[0], newpos[1] + 15))
                up()
                goto(newpos)
                write(item, 1)
            elif type(item) is list:
                draw_tree(item, newpos)
