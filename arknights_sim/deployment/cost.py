import math


def deployment_cost(base, count):
    return math.floor(base * (1 if count == 0 else 1.5 if count == 1 else 2))


def retreat_refund(paid):
    return math.floor(paid / 2)
