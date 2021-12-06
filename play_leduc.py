import random
import re

from numpy.random import choice


def normalize(d):
    normalizer = sum(d.values())
    return [(key, value / normalizer) for key, value in d.items()]


def get_action_probs(seq, strat, actions, prefix):
    return {
        action: strat[(seq, prefix + action)]
        for action in actions if (seq, prefix + action) in strat
    }


def play_kuhn(p1_strategy, p2_strategy, seed, trials):
    standing_match = 0
    random.seed(seed)
    cards = ['K', 'Q', 'J']
    random.shuffle(cards)
    p1_seq = f'/C:{cards[0]}?'
    p2_seq = f'/C:?{cards[1]}'

    actions = ['c', 'f', 'r']

    # We rely on the fact that strategies only contain valid actions as keys to check if what we did was valid.
    def play_betting_round(p1_seq, p2_seq):
        p1_actions_1 = normalize(
            get_action_probs(p1_seq, p1_strategy, actions, 'P1:'))
        p1_action_1 = choice([action for action, _ in p1_actions_1], 1,
                             [prob for _, prob in p1_actions_1])
        if p1_action_1 == 'f':
            return spend[0]
        elif p1_action_1 == 'c':
            assert p1_action_1 == 'c'
        else:
            assert re.fullmatch(p1_action_1, r'raise.*')
            standing_match = raise_val
            spend[0] -= raise_val
        p1_seq += f'/P1:{p1_action_1_1}'
        p2_seq += f'/P1:{p1_action_1_1}'

        p2_actions_1 = normalize(get_action_probs(p2_seq, p2_strategy,
                                                  actions))
        p2_action_1 = choice([action for action, _ in p2_actions_1], 1,
                             [prob for _, prob in p2_actions_1])
        if p2_action_1 == 'f':
            return -1 * spend[1]
        elif p2_action_1 == 'raise2':
            spend[1] -= (2 + standing_match)
            standing_match = 2
        else:
            assert p2_action_1 == 'c'
            standing_match = 0
            spend[1] -= standing_match
        p1_seq += f'/P2:{p2_action_1}'
        p2_seq += f'/P2:{p2_action_1}'
        if p2_action_1 == 'raise2':
            p1_actions_1_x = normalize(
                get_action_probs(p1_seq, p1_strategy, actions))
            p1_action_1_x = choice([action for action, _ in p1_actions_1_x], 1,
                                   [prob for _, prob in p1_actions_1_x])
            if p1_action_1_x == 'f':
                return spend[0]
            elif p2_action_1 == 'raise2':
                spend[1] -= (2 + standing_match)
                standing_match = 2
            else:
                assert p1_action_1_x == 'c'
                standing_match = 0
                spend[0] -= 2
            p1_seq += f'/P2:{p2_action_1}'
