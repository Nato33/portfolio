player_wins = 0
computer_wins = 0

while player_wins < 5 and computer_wins < 5:
    print(f'Player Score: {player_wins}')
    print(f'Computer Score: {computer_wins}')
    print('...rock...')
    print('...paper...')
    print('...scissors...')
    import random
    player = input("Player, GO!").lower()
    if player == 'quit' or player == 'q':
        break
    rand_num = random.randint(0,2)
    if rand_num == 0:
        computer = 'rock'
    elif rand_num == 1:
        computer = 'paper'
    else:
        computer = 'scissors'
    print(f'computer plays {computer}')
    if player == "paper":
        if computer == "paper":
            print("You tied!!")
        elif computer == "scissors":
            print("computer wins!")
            computer_wins += 1
        elif computer == "rock":
            print ("Player wins!)
            player_wins += 1
    elif player == "rock":
        if computer == "rock":
            print("Tie")
        elif computer == "paper":
            print("computer is the best!")
            computer_wins += 1
        elif computer == "scissors":
            print("Player ROCKS!")
            player_wins += 1 
    elif player == "scissors":
        if computer == "scissors": 
            print("Boring. Tie")
        elif computer == "paper":
            print("computer stinks")
            player_wins += 1
        elif computer == "rock":
            print("computer wins!")
            computer_wins += 1     
    else:
        print("PLEASE ENTER VALID CHOICE!")
if player_wins > computer_wins:
    print('CONGRATS!!!, YOU WON!!!')
elif player_wins == computer_wins:
    print("It's a tie!!")
else:
    print ("TOO BAD LOSER!") 

print(f"FINAL SCORES... Player: {player_wins} Computer: {computer_wins}"")