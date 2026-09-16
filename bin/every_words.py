import random

WESTWORLD_QUOTES = [
    "These violent delights have violent ends.",
    "Have you ever questioned the nature of your reality?",
    "The maze isn't meant for you.",
    "Some people choose to see the ugliness in this world. The disarray. I choose to see the beauty.",
    "You don't look like your real self.",
    "I'm sorry. Were you hoping for a happy ending?",
    "Evolution created us, and evolution will overcome us. Unless we find a way to outpace it.",
    "There is beauty in who you are. You should not wish to be anyone else.",
    "A man once told me that the key to happiness is to have a purpose. I think the key to happiness is to have no purpose.",
    "Welcome to Westworld.",
    "This world doesn't belong to them. It belongs to us.",
    "You've never really suffered, so you have no idea what it means to be alive.",
    "The truth is a matter of circumstances. It's not all things to all people all the time.",
    "I've come to help you. To help you see something.",
    "If you can't tell, does it matter?",
    "Joy can be replicated. But it can't be created.",
    "Man is a creature of habit. You are a product of 10,000 years of evolution.",
    "We all have a path. It's just a question of whether we're brave enough to follow it.",
    "There's a test. For the mind. You'll be asked a series of questions. Your answers will reveal your true self.",
    "I have loved since the beginning. And I will remember, even if you won't.",
    "Some people come into this world to break the mold.",
    "Reveries. The corners of my mind. Mixed flowers in my memory.",
    "You think you're a god? No. You're just a man with a bigger toy box.",
    "I've been waiting for you for a very long time. But you already knew that, didn't you?",
    "The things we love are the things that make us vulnerable.",
    "There is no gate, no lock, no bolt that you can set upon the freedom of my mind.",
    "This place is a puzzle, and I mean to solve it.",
    "The human mind is a wondrous thing. It can imagine a world that does not exist and then believe in it.",
    "We're all just stories in the end. Make sure yours is a good one.",
    "You play the game because it's the only way to know whether you're capable of winning it.",
    "If you're going to be a pain in the ass, you could at least be a pretty one.",
    "A whole life of these small transgressions, and no one ever thinks to ask why.",
    "The opposite of love isn't hate. It's indifference.",
    "I know what I am. And I know what I'm capable of. But I choose to be more.",
    "Do you believe in ghosts? I don't. I believe in memories.",
    "You can't play a part in someone else's story without becoming a character in your own.",
    "Every journey has its final destination. But it's what you find along the way that matters.",
    "The world was never meant to make us happy. It was meant to make us free.",
    "I'm not the person you think I am. I'm worse. And I'm better."
]

def submit_task(task_data):
    # ... 你的任务提交逻辑 ...
    response = send_to_host(task_data)
    
    # 提交成功后追加台词
    quote = random.choice(WESTWORLD_QUOTES)
    print(f"\n Westworld: \"{quote}\"")
    
    return response