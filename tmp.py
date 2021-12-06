'''
def get_action_val(action):
    if action == 'c':
        inc = 0
    elif action == 'raise2':
        inc = 2
    elif action == 'raise4':
        inc = 4
    return inc

def get_current_pot_and_chance(p1_seq, p2_seq)

    assert len(p1_seq) > len(p2_seq) or len(p2_seq) > len(p1_seq)

    longer_seq = p1_seq if len(p1_seq) > len(p2_seq) else p2_seq
    pot = [1, 1] # start w/ ante of 1 each
    looming_match = 0
    chance_card = None
    for event in longer_seq:
        res1 = re.fullmatch(r'P1:(.*)', event)
        res2 = re.fullmatch(r'P2:(.*)', event)
        res3 = re.fullmatch(r'C:(.*)', event)
        if res1:
            action = res1.group(1)
            inc = get_action_val(action)
            pot[0] += looming_match + inc
            looming_match = inc
        elif res2:
            action = res2.group(1)
            inc = get_action_val(action)
            pot[1] += looming_match + inc
            looming_match = inc
        elif res3:
            # Get public chance card
