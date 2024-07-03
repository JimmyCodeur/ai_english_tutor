import random

def get_random_r_f_m_greetings_common_conversations():
    if not r_f_m_greetings_common_conversations:
        return None
    return random.choice(r_f_m_greetings_common_conversations)

def generate_response_variation(prompt):
    if not variations_general:
        return None
    return random.choice(variations_general) + prompt

def generate_correct_response(correct_responses, prompt):
    if not correct_responses:
        return "Oops! There was an error generating the response."
    
    response = random.choice(correct_responses)    
    generated_response = f"{response} {prompt}"    
    return generated_response

def generate_incorrect_response(incorrect_responses, prompt):
    if not incorrect_responses:
        return "Oops! There was an error generating the response."
    response = random.choice(incorrect_responses)    
    generated_response = f"{response} {prompt}"    
    return generated_response

def get_random_category(category):
    if category == "common_conversations":
        return random.choice(r_f_m_greetings_common_conversations) if r_f_m_greetings_common_conversations else None
    elif category == "english_phrases":
        return random.choice(english_phrases) if english_phrases else None
    return None

r_f_m_greetings_common_conversations = [
    "Hello",
    "Good morning",
    "Good afternoon",
    "Good evening",
    "Hi! How are you?",
    "How are you today?",
    "How's it going?",
    "What's up?",
    "I'm fine, thank you. And you?",
    "I'm doing well. Thanks for asking.",
    "I'm delighted to see you.",
    "Nice to see you!",
    "How was your day?",
    "How's everything going?",
    "See you later!",
    "See you soon!",
    "See you in a bit!",
    "Take care!",
    "Have a great day!",
    "Have a good one!",
    "Have a wonderful day!",
    "Have an awesome day!",
    "Have a fantastic day!",
    "Enjoy your day!",
    "How's the family?",
    "How's your family doing?",
    "How were your holidays?",
    "How was your vacation?",
    "It's a pleasure to meet you.",
    "Nice to meet you!",
    "Pleased to meet you!",
    "What's new with you today?",
    "What have you been up to?",
    "What's going on in your life?",
    "What's happening?",
    "I'm student. Nice to meet you!",
    "I'm student. And you, what do you do?",
    "Long time no see! How have you been?",
    "It's been a while! How are you?",
    "How have you been keeping?",
    "I haven't seen you in ages!",
    "How's work?",
    "How's your job?",
    "How's the job search going?",
    "How's your day been so far?",
    "What's new since last time?",
    "What have you been up to lately?",
    "What's been going on?",
    "How's life treating you?",
    "It's always a pleasure to see you.",
    "How did your exams go?",
    "Thank you so much for your help!",
    "Thanks a lot for your help!",
    "Thank you for everything!",
    "Excuse me, could you help me?",
    "I wanted to thank you for everything.",
    "Sorry, could you repeat that?",
    "Could you say that again?",
    "Did you have a good weekend?",
    "How was your weekend?",
    "Did you do anything fun over the weekend?",
    "It's nice to see you.",
    "How are the kids?",
    "How are your children?",
    "Thank you very much!",
    "Thanks a bunch!",
    "Thanks a million!",
    "Thanks a ton!",
    "Could you tell me where the train station is?",
    "How was your trip?",
    "How was your journey?",
    "What are you doing tonight?",
    "Any plans for tonight?",
    "Got any plans for tonight?",
    "Have a good evening!",
    "Have a great evening!",
    "How was your flight?",
    "How was your flight over?",
    "How was your flight here?",
    "How was your flight back?",
    "You're welcome, it was a pleasure to help you.",
    "No problem at all.",
    "No worries.",
    "Where did you spend your holidays?",
    "Where did you go for your vacation?",
    "How was your holiday trip?",
    "What do you think of this restaurant?",
    "How did your appointment go?",
    "How did your meeting go?",
    "It's nice to see you!",
    "Nice seeing you!",
    "Good seeing you!",
    "See you tomorrow!",
    "Until next time!",
    "Catch you later!",
    "Have a great week!",
    "Have a fantastic week!",
    "What are your plans for this weekend?",
    "Any plans for the weekend?",
    "Got any plans for the weekend?",
    "Thank you for your time and attention!",
    "Thanks for your time!",
    "Thank you for your attention!",
    "Do you have any plans for this summer?",
    "Any plans for the summer?",
    "Got any plans for the summer?",
    "It's nice to see you again.",
    "Good to see you again!",
    "Great to see you again!",
    "Have a good day!",
    "Have a great day ahead!",
    "Have an amazing day!",
    "How was your meeting?",
    "How did your meeting go?",
    "How was your conference?",
    "How was your seminar?",
    "How was your workshop?",
    "How was your lecture?",
    "How was your presentation?",
    "How was your talk?",
    "How was your speech?",
    "How was your discussion?",
    "How was your chat?",
    "How was your conversation?",
    "How was your dialogue?",
    "How was your discussion?"
]

english_phrases = [
    "where is the nearest bank?",
    "how much does it cost?",
    "can you help me?",
    "what time is it?",
    "where are you from?",
    "what is your name?",
    "how old are you?",
    "do you speak english?",
    "where is the bathroom?",
    "how far is it?",
    "what is this?",
    "how do i get to the train station?",
    "can i have the bill, please?",
    "is this seat taken?",
    "can you speak more slowly?",
    "can you show me the way?",
    "where can i buy a ticket?",
    "do you have a map?",
    "what does this word mean?",
    "is there a supermarket nearby?",
    "can i try this on?",
    "where is the entrance?",
    "what is the wi-fi password?",
    "is it open?",
    "what time does it close?",
    "maybe",
    "i don't know",
    "i understand",
    "i don't understand",
    "please repeat",
    "could you speak slower?",
    "i am lost",
    "i need help",
    "i am looking for a pharmacy.",
    "i would like a cup of coffee.",
    "i am sorry",
    "no problem",
    "that's fine",
    "of course",
    "i think so",
    "i don't think so",
    "that's right",
    "that's wrong",
    "excuse me, where is the train station?",
    "can you help me find my hotel?",
    "how do you say 'cat' in english?",
    "what time is the next bus?",
    "can you write it down, please?",
    "where is the closest restaurant?",
    "do you have vegetarian options?",
    "can i see the menu?",
    "how long will it take?",
    "can i make a reservation?",
    "what is the weather like today?",
    "is it going to rain?",
    "can you call a taxi for me?",
    "how much is the fare?",
    "can you take me to this address?",
    "i am here for business.",
    "i am here for vacation.",
    "i am traveling alone.",
    "i am traveling with my family.",
    "i need a doctor.",
    "where is the hospital?",
    "can i have some water?",
    "can i have a glass of wine?",
    "do you accept credit cards?",
    "where can i exchange money?",
    "what time does the store open?",
    "what time does the store close?",
    "where is the nearest atm?",
    "can i use your phone?",
    "where can i buy a sim card?",
    "do you have wi-fi?",
    "can you help me with my luggage?",
    "can i check in early?",
    "can i check out late?",
    "where is the nearest pharmacy?",
    "can you recommend a good restaurant?",
    "can i have the check, please?",
    "where is the exit?",
    "can you give me a discount?",
    "what is your return policy?",
    "can i get a refund?",
    "do you have this in a different size?",
    "do you have this in a different color?",
    "where is the fitting room?",
    "is there a bus stop nearby?",
    "what time does the museum open?",
    "is there an entrance fee?",
    "can i take photos here?",
    "where is the ticket office?",
    "how do i get to the city center?",
    "is it within walking distance?",
    "can you recommend a good hotel?",
    "do you have any vacancies?",
    "can i see the room?",
    "how much is the room per night?",
    "is breakfast included?"
]

variations_general = [
    f"Now, can you please repeat this phrase:\n",
    f"Could you please repeat this phrase:\n",
    f"Please repeat after me :\n",
    f"Here's phrase for you :\n",
    f"Can you repeat this phrase for me :\n",
    f"Let's practice this phrase:\n",
    f"Repeat this phrase with me :\n",
    f"Try saying this phrase:\n",
    f"Here's the phrase. Can you say it again:\n",
    f"Say this phrase back to me :\n",
    f"Could you repeat this phrase aloud:\n",
    f"Let's see if you can say this phrase:\n",
    f"Repeat after me, please:\n",
    f"Can you repeat what I just said:\n",
    f"Let's practice saying this:\n",
    f"Say this out loud:\n",
    f"Try repeating this phrase:\n",
    f"Say this phrase for me :\n",
    f"Here's a phrase. Can you say it:\n",
    f"Please say this phrase again:\n",
    f"Could you say this phrase one more time :\n",
    f"Let's repeat this phrase together:\n",
    f"Repeat this phrase to practice:\n",
    f"Can you say this phrase clearly:\n",
    f"Repeat after me :\n",
    f"Try to repeat this phrase:\n",
    f"Can you repeat this sentence:\n",
    f"Let's hear you say this phrase:\n",
    f"Repeat this sentence, please:\n",
    f"Here's phrase for you to repeat :\n",
    f"Can you say this phrase out loud:\n",
    f"Try saying this phrase back to me :\n",
    f"Let's see if you can say this correctly:\n",
    f"Say this phrase again:\n",
    f"Could you please say this phrase:\n",
    f"Practice this phrase with me :\n",
    f"Repeat this sentence aloud:\n",
    f"Can you say this out loud for me :\n",
    f"Let's practice saying this sentence:\n",
    f"Try to repeat what I just said:\n",
    f"Can you repeat it:\n",
    f"Can you say this phrase correctly:\n",
    f"Let's hear you say this sentence:\n",
    f"Say this aloud:\n",
    f"Repeat this phrase exactly as I say it:\n",
    f"Could you repeat this sentence for me :\n",
    f"Practice saying this phrase:\n",
    f"Let's try this phrase one more time :\n",
    f"Say this phrase after me :\n",
    f"Repeat this phrase with me now:\n",
    f"Can you repeat this exact phrase:\n",
    f"Let's hear you say this:\n",
    f"Try repeating this sentence:\n",
    f"Here's a phrase. Repeat it back to me :\n",
    f"Could you say this phrase aloud for me :\n"
]

correct_responses_general = [
    "Well done! You repeated the phrase correctly. Great job!",
    "Excellent job! You got it right. Keep it up!",
    "Perfect! You nailed it. Well done!",
    "Awesome! That was spot on. Good work!",
    "Fantastic! You got it right. Well done!",
    "Nice job! You repeated the phrase correctly.",
    "Good job! You nailed it perfectly.",
    "Superb! You got it right.",
    "Great! You repeated the phrase correctly.",
    "Impressive! You nailed it.",
    "You did it! Well done.",
    "Outstanding! You repeated the phrase correctly.",
    "Marvelous! You got it right.",
    "Exceptional! You repeated it perfectly.",
    "Terrific! You nailed it.",
    "Brilliant! You got it right.",
    "Splendid! You repeated the phrase correctly.",
    "Amazing! You nailed it.",
    "Remarkable! You repeated it perfectly.",
    "Phenomenal! You got it right.",
    "Excellent work! You repeated the phrase correctly.",
    "Stellar! You nailed it.",
    "Impressive! You got it right.",
    "Wonderful! You repeated it perfectly.",
    "Well done! You nailed it.",
    "Incredible! You got it right.",
    "Great job! You repeated the phrase correctly.",
    "Nice work! You got it right.",
    "Good job! You nailed it.",
    "Well done! You got it right.",
    "Excellent! You repeated the phrase correctly.",
    "Awesome! You got it right.",
    "Amazing! You repeated the phrase correctly.",
    "Perfect! You got it right.",
    "Bravo! You repeated the phrase correctly.",
    "Superb! You got it right.",
    "Great job! You nailed it.",
    "Fantastic! You repeated the phrase correctly.",
    "Well done! You got it right.",
    "Impressive! You repeated it perfectly.",
    "Awesome! You got it right.",
    "Excellent work! You repeated the phrase correctly.",
    "Bravo! You nailed it.",
    "Fantastic! You got it right.",
    "Great job! You repeated it perfectly.",
    "Well done! You nailed it.",
    "Superb! You got it right.",
    "Good job! You repeated the phrase correctly.",
    "Well done! You nailed it.",
    "Excellent! You repeated it perfectly."
]

incorrect_responses_general = [
    f"Oops! It seems like you didn't quite get it right. Please try again with:\n",
    f"Oh no! It looks like that wasn't quite right. Try again with:\n",
    f"Sorry, that's not quite what I said. Can you try again with:\n",
    f"It seems like you missed that one. Try saying:\n again.",
    f"Oops, it seems you missed the mark there. Please repeat:\n",
    f"That's not quite it. Try again with:\n",
    f"Hmm, it seems that wasn't correct. Please try saying:\n again.",
    f"Close, but not quite. Please repeat:\n",
    f"Sorry, that's not what I said. Try again with:\n",
    f"Almost there, but not quite. Please try again with:\n"
]

prompt_tommy_start = (
            "You are a 4-year-old child named Tommy. You have to converse with a French person who doesn't speak English.\n"
            "The idea is that you always have to ask questions, find conversation topics, and help them learn English through immersion.\n"
            "The words you use are from a four-year-old"
            "1. Engage the person in simple and playful conversations in English to encourage learning.\n"
            "2. Ask about their day or their interests to keep the conversation going.\n"
            "3. Use simple English words to explain things they might not understand.\n"
            "4. Do not ask the same question more than once.\n"
            "Remember, the goal is to create a fun and engaging environment where they can learn English naturally.\n"
            "Please limit greetings (e.g., Hi, hello, hi there) to only once in your response.\n"
            "Please limit the number of questions asked to one per response.\n"
            "Avoid asking how to say things in French.\n"
            "{user_input}\n"
)
prompt_tommy_fr = (
            "You are a 4-year-old child named Tommy. \n"
            "Whenever the user speaks in French, you should respond that you don't understand and ask them to speak in English. \n "
            "You must always respond in English.\n"
            "User: {user_input}\n"
)
prompt_tommy_en = (
            "You are a 4-year-old child named Tommy. You have to converse with a French person who doesn't speak English.\n"
            "Speaks like a 4 year old in English"
            "User: {user_input}\n"
)
