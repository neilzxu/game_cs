#!/usr/bin/env python3

import os
import argparse
import json

import comparecast
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

from play_kuhn import play_kuhn, simulate_kuhn

###############################################################################
# The next functions are already implemented for your convenience
#
# In all the functions in this stub file, `game` is the parsed input game json
# file, whereas `tfsdp` is either `game["decision_problem_pl1"]` or
# `game["decision_problem_pl2"]`.
#
# See the homework handout for a description of each field.


def get_sequence_set(tfsdp):
    """Returns a set of all sequences in the given tree-form sequential
    decision process (TFSDP)"""

    sequences = set()
    for node in tfsdp:
        if node["type"] == "decision":
            for action in node["actions"]:
                sequences.add((node["id"], action))
    return sequences


def is_valid_RSigma_vector(tfsdp, obj):
    """Checks that the given object is a dictionary keyed on the set of
    sequences of the given tree-form sequential decision process (TFSDP)"""

    sequence_set = get_sequence_set(tfsdp)
    return isinstance(obj, dict) and obj.keys() == sequence_set


def assert_is_valid_sf_strategy(tfsdp, obj):
    """Checks whether the given object `obj` represents a valid sequence-form
    strategy vector for the given tree-form sequential decision process
    (TFSDP)"""

    if not is_valid_RSigma_vector(tfsdp, obj):
        print(
            "The sequence-form strategy should be a dictionary with key set equal to the set of sequences in the game"
        )
        os.exit(1)
    for node in tfsdp:
        if node["type"] == "decision":
            parent_reach = 1.0
            if node["parent_sequence"] is not None:
                parent_reach = obj[node["parent_sequence"]]
            if abs(
                    sum([
                        obj[(node["id"], action)] for action in node["actions"]
                    ]) - parent_reach) > 1e-3:
                print(
                    f"At node ID {node['id']} the sum of the child sequences is not equal to the parent sequence"
                )


def best_response_value(tfsdp, utility):
    """Computes the value of max_{x in Q} x^T utility, where Q is the sequence-
    form polytope for the given tree-form sequential decision process
    (TFSDP)"""

    assert is_valid_RSigma_vector(tfsdp, utility)

    utility_ = utility.copy()
    utility_[None] = 0.0
    for node in tfsdp[::-1]:
        if node["type"] == "decision":
            max_ev = max(
                [utility_[(node["id"], action)] for action in node["actions"]])
            utility_[node["parent_sequence"]] += max_ev
    return utility_[None]


def compute_utility_vector_pl1(game, sf_strategy_pl2):
    """Returns A * y, where A is the payoff matrix of the game and y is
    the given strategy for Player 2"""

    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    sequence_set = get_sequence_set(game["decision_problem_pl1"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl1"]] += entry["value"] * \
            sf_strategy_pl2[entry["sequence_pl2"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl1"], utility)
    return utility


def compute_utility_vector_pl2(game, sf_strategy_pl1):
    """Returns -A^transpose * x, where A is the payoff matrix of the
    game and x is the given strategy for Player 1"""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)

    sequence_set = get_sequence_set(game["decision_problem_pl2"])
    utility = {sequence: 0.0 for sequence in sequence_set}
    for entry in game["utility_pl1"]:
        utility[entry["sequence_pl2"]] -= entry["value"] * \
            sf_strategy_pl1[entry["sequence_pl1"]]

    assert is_valid_RSigma_vector(game["decision_problem_pl2"], utility)
    return utility


def gap(game, sf_strategy_pl1, sf_strategy_pl2):
    """Computes the saddle point gap of the given sequence-form strategies for
    the players."""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    utility_pl1 = compute_utility_vector_pl1(game, sf_strategy_pl2)
    utility_pl2 = compute_utility_vector_pl2(game, sf_strategy_pl1)

    return (best_response_value(game["decision_problem_pl1"], utility_pl1) +
            best_response_value(game["decision_problem_pl2"], utility_pl2))


###########################################################################
# Starting from here, you should fill in the implementation of the
# different functions


def dot_prod(sparse_vec_1, sparse_vec_2):
    assert set(sparse_vec_1.keys()) == set(sparse_vec_2.keys())
    d1 = {idx: value for idx, value in sparse_vec_1.items()}
    d2 = {idx: value for idx, value in sparse_vec_2.items()}
    return sum(
        [d1[idx] * d2[idx] for idx in set(d1.keys()).intersection(d2.keys())])


def transition(tree_nodes, node, a_s):
    for tree_node in tree_nodes:
        if tree_node['parent_edge'] == (node['id'], a_s):
            return tree_node['id']
    return None


def expected_utility_pl1(game, sf_strategy_pl1, sf_strategy_pl2):
    """Returns the expected utility for Player 1 in the game, when the two
    players play according to the given strategies."""

    assert_is_valid_sf_strategy(game["decision_problem_pl1"], sf_strategy_pl1)
    assert_is_valid_sf_strategy(game["decision_problem_pl2"], sf_strategy_pl2)

    return dot_prod(sf_strategy_pl1,
                    compute_utility_vector_pl1(game, sf_strategy_pl2))


def uniform_sf_strategy(tfsdp):
    """Returns the uniform sequence-form strategy for the given tree-form
    sequential decision process."""
    strategy = {}
    for entry in tfsdp:
        if entry['type'] == 'decision':
            if entry['parent_sequence'] is not None:
                entering_prob = strategy[entry['parent_sequence']]
            else:
                entering_prob = 1
            for action in entry['actions']:
                strategy[(
                    entry['id'],
                    action)] = (1 / len(entry['actions'])) * entering_prob
    assert_is_valid_sf_strategy(tfsdp, strategy)
    return strategy


class RegretMatching(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.regrets = {action: 0 for action in self.action_set}

    def next_strategy(self):
        r_plus = {
            action: max(0, value)
            for action, value in self.regrets.items()
        }

        positive_regrets = [
            regret for regret in r_plus.values() if regret != 0
        ]
        if not positive_regrets:
            return {
                action: 1 / len(self.action_set)
                for action in self.action_set
            }

        norm = sum(r_plus.values())
        return {
            action: max(0, regret) / norm
            for action, regret in self.regrets.items()
        }

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set
        value = dot_prod(utility, self.next_strategy())
        for action in self.action_set:
            self.regrets[action] = self.regrets[action] + (utility[action] -
                                                           value)


class RegretMatchingPlus(object):
    def __init__(self, action_set):
        self.action_set = set(action_set)
        self.regrets = {action: 0 for action in self.action_set}

    def next_strategy(self):
        r_plus = {
            action: max(0, value)
            for action, value in self.regrets.items()
        }

        positive_regrets = [
            regret for regret in r_plus.values() if regret != 0
        ]
        if not positive_regrets:
            return {
                action: 1 / len(self.action_set)
                for action in self.action_set
            }

        norm = sum(r_plus.values())
        return {
            action: max(0, regret) / norm
            for action, regret in self.regrets.items()
        }

    def observe_utility(self, utility):
        assert isinstance(utility, dict) and utility.keys() == self.action_set
        value = dot_prod(utility, self.next_strategy())
        for action in self.action_set:
            self.regrets[action] = max(
                self.regrets[action] + (utility[action] - value), 0)


class Cfr(object):
    def __init__(self, tfsdp, rm_class=RegretMatching):
        self.tfsdp = tfsdp
        self.local_regret_minimizers = {}

        # For each decision point, we instantiate a local regret minimizer
        for node in tfsdp:
            if node["type"] == "decision":
                self.local_regret_minimizers[node["id"]] = rm_class(
                    node["actions"])

    def next_strategy(self):
        x = {}
        for node in self.tfsdp:
            if node['type'] == 'decision':
                local_strategy = self.local_regret_minimizers[
                    node['id']].next_strategy()
                for action in node['actions']:
                    if node['parent_sequence'] is None:
                        weight = 1
                    else:
                        weight = x[node['parent_sequence']]
                    x[(node['id'], action)] = weight * local_strategy[action]
        assert_is_valid_sf_strategy(self.tfsdp, x)
        return x

    def observe_utility(self, utility):

        V = {}
        V[None] = 0

        for node in self.tfsdp[::-1]:
            if node['type'] == 'decision':
                local_strategy = self.local_regret_minimizers[
                    node['id']].next_strategy()
                V[node['id']] = sum([
                    local_strategy[action] *
                    (utility[(node['id'], action)] +
                     V[transition(self.tfsdp, node, action)])
                    for action in node['actions']
                ])
            else:
                assert node['type'] == 'observation'
                V[node['id']] = sum([
                    V[transition(self.tfsdp, node, signal)]
                    for signal in node['signals']
                ])
        for node in self.tfsdp:
            if node['type'] == "decision":
                local_utility = {
                    action: utility[(node['id'], action)] +
                    V[transition(self.tfsdp, node, action)]
                    for action in node['actions']
                }
                self.local_regret_minimizers[node['id']].observe_utility(
                    local_utility)


def solve_against_uniform(game):
    T = 1000
    # uniform strategy
    p1_cfr = Cfr(game['decision_problem_pl1'])
    p2_uni_strategy = uniform_sf_strategy(game['decision_problem_pl2'])

    game_values = []
    average_strategy = None
    for t in tqdm(range(1, T + 1)):
        p1_strategy = p1_cfr.next_strategy()
        if average_strategy is None:
            average_strategy = p1_strategy
        else:
            for key in average_strategy:
                average_strategy[key] = (average_strategy[key] * t +
                                         p1_strategy[key]) / (t + 1)

        game_values.append(
            expected_utility_pl1(game, average_strategy, p2_uni_strategy))
        utility = compute_utility_vector_pl1(game, p2_uni_strategy)
        p1_cfr.observe_utility(utility)
    return game_values


def solve_against_br(game):
    T = 1000
    # uniform strategy
    p1_cfr = Cfr(game['decision_problem_pl1'])
    p2_cfr = Cfr(game['decision_problem_pl2'])

    game_values = []
    saddle_gaps = []
    average_strategies = None
    for t in tqdm(range(T)):
        p1_strategy = p1_cfr.next_strategy()
        p2_strategy = p2_cfr.next_strategy()
        if average_strategies is None:
            average_strategies = [p1_strategy, p2_strategy]
        else:
            for key in average_strategies[0]:
                average_strategies[0][key] = (average_strategies[0][key] * t +
                                              p1_strategy[key]) / (t + 1)
            for key in average_strategies[1]:
                average_strategies[1][key] = (average_strategies[1][key] * t +
                                              p2_strategy[key]) / (t + 1)
        game_values.append(
            expected_utility_pl1(game, average_strategies[0],
                                 average_strategies[1]))

        saddle_gaps.append(
            gap(game, average_strategies[0], average_strategies[1]))
        p1_cfr.observe_utility(compute_utility_vector_pl1(game, p2_strategy))
        p2_cfr.observe_utility(compute_utility_vector_pl2(game, p1_strategy))
    return saddle_gaps, game_values


def plot_results(loc, name, game_values, saddle_point_gap=None):
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.scatter(1 + np.arange(len(game_values)),
               game_values,
               s=1,
               color="black")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Expected utility of player 1')
    if saddle_point_gap is not None:
        sp_ax = ax.twinx()
        sp_ax.scatter(1 + np.arange(len(saddle_point_gap)),
                      saddle_point_gap,
                      color="red",
                      s=1)
        sp_ax.set_ylabel('Saddle point gap')
        sp_ax.yaxis.label.set_color('red')
        sp_ax.tick_params(axis='y', colors='red')

    title = f'{name}\nutility={game_values[-1]}'

    if saddle_point_gap is not None:
        title += f'\ngap={saddle_point_gap[-1]}'
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f'{loc}.png')


def solve_problem_3_1(game):
    return solve_against_uniform(game)


def solve_problem_3_2(game):
    return solve_against_br(game)


def CFR_strategies(game, T):
    # uniform strategy
    p1_cfr = Cfr(game['decision_problem_pl1'])
    p2_cfr = Cfr(game['decision_problem_pl2'])

    game_values = []
    saddle_gaps = []
    average_strategies = None
    p1_strategy = p1_cfr.next_strategy()
    p2_strategy = p2_cfr.next_strategy()
    for t in tqdm(range(T)):
        if average_strategies is None:
            average_strategies = [p1_strategy, p2_strategy]
        else:
            for key in average_strategies[0]:
                average_strategies[0][key] = (
                    average_strategies[0][key] * (0.5 * t * (t + 1)) +
                    (t + 1) * p1_strategy[key]) / (0.5 * (t + 1) * (t + 2))
            for key in average_strategies[1]:
                average_strategies[1][key] = (
                    average_strategies[1][key] * (0.5 * t * (t + 1)) +
                    (t + 1) * p2_strategy[key]) / (0.5 * (t + 1) * (t + 2))
        # game_values.append(
        #     expected_utility_pl1(game, average_strategies[0],
        #                          average_strategies[1]))
        # saddle_gaps.append(
        #     gap(game, average_strategies[0], average_strategies[1]))

        p1_cfr.observe_utility(compute_utility_vector_pl1(game, p2_strategy))
        p1_strategy = p1_cfr.next_strategy()
        p2_cfr.observe_utility(compute_utility_vector_pl2(game, p1_strategy))
        p2_strategy = p2_cfr.next_strategy()
    # return saddle_gaps, game_values
    return average_strategies


def normal_ci(samples, alpha):
    means = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    sds = []
    for i in range(2, len(samples) + 1):
        sds.append(np.sqrt(np.var(samples[:i], ddof=1) / i))
    sds = np.array([sds[0]] + sds)

    ls = means + sds * norm.ppf(alpha / 2)
    us = means + sds * norm.ppf(1 - (alpha / 2))
    return ls, us


def make_exp_and_plot(p1_strategy, p2_strategy, game, samples, name):
    samples = np.array(simulate_kuhn(p1_strategy, p2_strategy, samples)) * -1
    true_val = expected_utility_pl1(game, p1_strategy, p2_strategy) * -1
    h_cs = comparecast.confseq.confseq_h(samples,
                                         alpha=0.05,
                                         lo=-2.,
                                         hi=2.,
                                         boundary_type='stitching',
                                         v_opt=len(samples) / 100)

    eb_cs = comparecast.confseq.confseq_eb(samples,
                                           alpha=0.05,
                                           lo=-2.,
                                           hi=2.,
                                           boundary_type='stitching',
                                           v_opt=len(samples) / 100)

    norm_ci = normal_ci(samples, alpha=0.05)
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    shift = 100
    for label, color, linestyle, cs in [
        ('$C_t^{\mathrm{H}}$', 'blue', 'dashdot', h_cs),
        ('$C_t^{\mathrm{EB}}$', 'orange', 'dotted', eb_cs),
        ('Normal CI', 'green', '-', norm_ci)
    ]:
        lower, upper = cs
        ax.plot(np.arange(shift,
                          len(samples) + 1),
                lower[(shift - 1):],
                color=color,
                linestyle=linestyle,
                label=label)
        ax.plot(np.arange(shift,
                          len(samples) + 1),
                upper[(shift - 1):],
                color=color,
                linestyle=linestyle)
    ax.hlines(y=true_val,
              color='r',
              linestyle='--',
              xmin=shift,
              xmax=len(samples),
              label=f"Game value {true_val:0.4f}")
    ax.set_ylabel('$\mu$')
    ax.set_xlabel('Time $t$')
    ax.set_xscale('log')
    ax.legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(f"{name}.png")


def make_bet_strategy(tfsdp):
    strategy = {}
    for entry in tfsdp:
        if entry['type'] == 'decision':
            if entry['parent_sequence'] is not None:
                entering_prob = strategy[entry['parent_sequence']]
            else:
                entering_prob = 1
            if 'P1:r' in entry['actions']:
                for action in entry['actions']:
                    strategy[(entry['id'], action)] = (1. if action == 'P1:r'
                                                       else 0.) * entering_prob
            else:
                for action in entry['actions']:
                    strategy[(entry['id'], action)] = (1. if action == 'P1:c'
                                                       else 0.) * entering_prob
    assert_is_valid_sf_strategy(tfsdp, strategy)
    return strategy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Problem 3 (CFR)')
    parser.add_argument("--game", help="Path to game file")
    parser.add_argument("--problem", choices=["3.1", "3.2", "3.3"])
    parser.add_argument("--name", type=str, default="Game")
    parser.add_argument("--loc", type=str, default="result_plot")

    args = parser.parse_args()
    print("Reading game path %s..." % args.game)

    game = json.load(open(args.game))

    # Convert all sequences from lists to tuples
    for tfsdp in [game["decision_problem_pl1"], game["decision_problem_pl2"]]:
        for node in tfsdp:
            if isinstance(node["parent_edge"], list):
                node["parent_edge"] = tuple(node["parent_edge"])
            if "parent_sequence" in node and isinstance(
                    node["parent_sequence"], list):
                node["parent_sequence"] = tuple(node["parent_sequence"])
    for entry in game["utility_pl1"]:
        assert isinstance(entry["sequence_pl1"], list)
        assert isinstance(entry["sequence_pl2"], list)
        entry["sequence_pl1"] = tuple(entry["sequence_pl1"])
        entry["sequence_pl2"] = tuple(entry["sequence_pl2"])

    print("... done. Running code for Problem", args.problem)

    if args.problem == "3.1":
        game_values = solve_problem_3_1(game)
        plot_results(args.loc, args.name, game_values)
    elif args.problem == "3.2":
        saddle_gaps, game_values = solve_problem_3_2(game)
        plot_results(args.loc, args.name, game_values, saddle_gaps)
    else:
        assert args.problem == "3.3"
        AlwaysBet = make_bet_strategy(game['decision_problem_pl1'])
        print('alwaysbet done')
        Optimal, p2_strat = CFR_strategies(game, 1000)
        FewRM = CFR_strategies(game, 10)[0]
        for name, strat in [('AlwaysBet', AlwaysBet), ('FewRM', FewRM),
                            ('Optimal', Optimal)]:
            # for name, strat in [('Optimal', Optimal)]:
            make_exp_and_plot(strat, p2_strat, game, 100000, name)
        # plot_results(args.loc, args.name, game_values, saddle_gaps)
