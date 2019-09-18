from random import shuffle
# Each instance of Card  should have a suit ("Hearts", "Diamonds", "Clubs", or "Spades").
# Each instance of Card  should have a value ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K").
# Card 's __repr__  method should display the card's value and suit (e.g. "A of Clubs", "J of Diamonds", etc.)


class Card:
	def __init__(self, value, suit):
		self.value = value
		self.suit = suit

	def __repr__(self):
		# return "{} of {}".format(self.value, self.suit)
		return f"{self.value} of {self.suit}"



class Deck:
	def __init__(self):
		suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
		values = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
		self.cards = [Card(value, suit) for suit in suits for value in values]

	def __repr__(self):
		return f"Deck of {self.count()} cards"

	def count(self):
		return len(self.cards)

	def _deal(self, num):
		count = self.count()
		actual = min([count,num])
		if count == 0:
			raise ValueError("All cards have been dealt")
		cards = self.cards[-actual:]
		self.cards = self.cards[:-actual]
		return cards

	def deal_card(self):
		return self._deal(1)[0]

	def deal_hand(self, hand_size):
		return self._deal(hand_size)

	def shuffle(self):
		if self.count() < 52:
			raise ValueError("Only full decks can be shuffled")
		shuffle(self.cards)


d = Deck()
d.shuffle()
card = d.deal_card()
print(card)
hand = d.deal_hand(50)
card2 = d.deal_card()
print(card2)
print(d.cards)


# print(d.cards)







