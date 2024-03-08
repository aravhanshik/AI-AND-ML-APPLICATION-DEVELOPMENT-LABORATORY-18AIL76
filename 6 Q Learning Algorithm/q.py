import numpy as np

terminal = 5

actions = ['UP','DW','LF','RG']

rws = np.array([-1]*6)
rws[5] = 10

paths = [(0, ['UP','UP','UP','RG']), (4, ['RG','RG','LF','UP'])]

alpha = 0.5
gamma = 0.8

def print_value(value):
    print('[' + str(value[2]) + ' ' + str(value[5]))
    print(str(value[1]) + ' ' + str(value[4]))
    print(str(value[0]) + ' ' + str(value[3]) + ']\n')
  
def update_value(value, state, action):
    index = actions.index(action)
    next_state = state
    rw = 0
    if action == 'UP':
        if state == 2 or state == 5:
            rw = -10
        else:
            next_state = state + 1
            
    elif action == 'DW':
        if state == 0 or state == 3:
            rw = -10
        else:
            next_state = state - 1

    elif action == 'LF':
        if state == 0 or state == 1 or state == 2:
            rw = -10
        else:
            next_state = state - 3

    elif action == 'RG':
        if state == 3 or state == 4 or state == 5:
            rw = -10
        else:
            next_state = state + 3
    if rw == 0:
        rw = rws[next_state]
    value[index][state] = value[index][state] + alpha * (rw + gamma * max(value[i][next_state] for i in range(4)) - value[index][state])
    return value, next_state
  
def return_policy(value):
    policy = np.array([' ']*6)
    policy[5] = '+10'
    for state in range(5):
        policy[state] = actions[np.argmax([value[action][state] for action in range(4)])]
    print(policy[2] + ' ' + policy[5])
    print(policy[1] + ' ' + policy[4])
    print(policy[0] + ' ' + policy[3]+ '\n')
  
def main():
    value = [np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)]
    for i in range(len(paths)):
        state = paths[i][0]
        actions = paths[i][1]
        for action in actions:
            value, state = update_value(value, state, action)
            if state == terminal:
                break
 
        print_value(value[0])
  
        print_value(value[1])

        print_value(value[2])

        print_value(value[3])

        return_policy(value)
      
if __name__ == '__main__':
    main()
