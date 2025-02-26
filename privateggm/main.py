'''
Program description:
    - This program aims to allow the user to craft a funny story using inputted words
    - The program asks for 14 variables, then generates a full story about an upcoming programmer
    - The program alters or combines certain inputs to add an element of surprise

Inputs:
    - User-provided words (nouns, verbs, adjectives, etc.)

Outputs:
    - A completed Mad Libs story based on user inputs

Author: Sky Cooper
Date: Feb. 15, 2025
'''

#  Introduction
print("Hi there! My name is Mr. Word Bot😊")
name = input("What's your name? ").capitalize()
print(f"Nice to meet you, {name}! Let's play Mad Libs!") #

#  Inputs
emotion_one = input("Enter an emotion: ")
adjective_ing = input("Enter an ajective ending in 'ing': ")
adjective_negative = input("Enter a negative adjective: ")
emotion_two = input("Enter an emotion: ")
number = input("Enter a whole number: ")
verb = input("Enter a verb: ")
noun_plural = input("Enter a plural noun: ")
noun_abstract = input("Enter an abstract noun: ")
adverb = input("Enter an adverb: ")
adjective_one = input("Enter an adjective: ")
adjective_two = input("Enter an adjective: ").capitalize()
animal = input("Enter an animal: ").capitalize()
number_decimal = input("Enter a decimal number: ")

#Conversions
hours = int(number)*24
library_name = adjective_two + animal

#Output
print()
print("Excellent! Here's your story:")
print()
print(f"Once upon a time, there was a student named {name} who was new to programming. At first, {name} felt {emotion_one} because they had never programmed before. The code seemed like a/an {adjective_ing} task and the {adjective_negative} errors were everywhere. \n\nOne day, while {name} was trying to program, they felt {emotion_two} and thought they might never get the hang of it. They had spent {hours} hours on their program with no progress. However, they decided to {verb} {noun_plural} to improve their skills. They understood that with {noun_abstract} and practice, they could overcome the challenges they were facing. \n\nAs {name} kept learning and practicing, they started to understand how to program {adverb}. The {adjective_one} errors became easier to fix, and {name} began to have fun creating {adjective_two} programs. They even created their own library called {library_name} in {number_decimal} hours. In the end, {name} realized that not only could they learn to program but also enjoy the process. They felt proud of how far they had come. The End.")
print()
print(f"Brilliant message {name}!".upper())