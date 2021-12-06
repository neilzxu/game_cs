import random

from numpy.random import choice


def normalize(d):
    normalizer = sum(d.values())
    return [(key, value / normalizer) for key, value in d.items()]


def get_action_probs(seq, strat, actions, prefix):
    # print(seq, prefix, actions)
    # print(strat)
    return {
        action: strat[(seq, f'{prefix}:{action}')]
        for action in actions if (seq, f'{prefix}:{action}') in strat
    }


def sample_action(seq, strat, actions, prefix):
    action_probs = normalize(get_action_probs(seq, strat, actions, prefix))
    return choice([action for action, _ in action_probs], 1,
                  [prob for _, prob in action_probs])[0]


def update_seq(p1_seq, p2_seq, action, prefix):
    return p1_seq + f'/{prefix}:{action}', p2_seq + f'/{prefix}:{action}'


def play_kuhn(p1_strategy, p2_strategy):
    cards = ['K', 'Q', 'J']
    values = {'K': 3, 'Q': 2, 'J': 1}

    random.shuffle(cards)
    showdown = 1 if values[cards[0]] > values[cards[1]] else -1
    p1_seq = f'/C:{cards[0]}?'
    p2_seq = f'/C:?{cards[1]}'

    actions = ['c', 'f', 'r']

    action = sample_action(p1_seq, p1_strategy, actions, 'P1')
    p1_seq, p2_seq = update_seq(p1_seq, p2_seq, action, 'P1')
    if action == 'c':
        action = sample_action(p2_seq, p2_strategy, actions, 'P2')
        p1_seq, p2_seq = update_seq(p1_seq, p2_seq, action, 'P2')
        if action == 'c':
            return showdown * 1
        else:
            assert action == 'r'
            action = sample_action(p1_seq, p1_strategy, actions, 'P1')
            p1_seq, p2_seq = update_seq(p1_seq, p2_seq, action, 'P1')
            if action == 'f':
                return -1
            else:
                assert action == 'c'
                return showdown * 2
    else:
        assert action == 'r'
        action = sample_action(p2_seq, p2_strategy, actions, 'P2')
        p1_seq, p2_seq = update_seq(p1_seq, p2_seq, action, 'P2')
        if action == 'f':
            return 1
        else:
            assert action == 'c'
            return showdown * 2


def simulate_kuhn(p1_strategy, p2_strategy, times):
    rewards = []
    for _ in range(times):
        rewards.append(play_kuhn(p1_strategy, p2_strategy))
    return rewards
