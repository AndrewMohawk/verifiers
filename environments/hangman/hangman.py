import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

def check_answer_reward_func(parser, completion, answer, **kwargs) -> float:
    guess = parser.parse_answer(completion)
    return 1.0 if guess == "[" + answer.upper() + "]" else 0.0


# def partial_credit_reward_func(parser, completion, **kwargs) -> float:
#     """Partial credit based on percentage of letters revealed"""
#     final_env_response = parser.get_user_messages(completion)[-1]["content"].strip()
#     # Parse final board state, count revealed vs total letters
#     # Look for lines with underscores and letters
#     return 0.0


# def letter_accuracy_reward_func(parser, completion, **kwargs) -> float:
#     """Accuracy of individual letter guesses (correct/total)"""
#     messages = parser.get_user_messages(completion)
#     correct_guesses = 0
#     total_guesses = 0
#     # Count "is in the word" vs "is not in the word" messages
#     return 0.0


# def count_turns_reward_func(parser, completion, answer, **kwargs) -> float:
#     """Higher score for solving in fewer turns"""
#     return 0.0


# def lives_remaining_reward_func(parser, completion, **kwargs) -> float:
#     """Bonus for preserving lives"""
#     final_env_response = parser.get_user_messages(completion)[-1]["content"].strip()
#     # Parse "tries left" or "lives left" from final message
#     return 0.0


# def vowel_strategy_reward_func(parser, completion, **kwargs) -> float:
#     """Reward for guessing vowels early"""
#     messages = parser.get_user_messages(completion)[:3]  # First 3 guesses
#     vowels_guessed = 0
#     # Check if A,E,I,O,U appear in first few guesses
#     return 0.0


# def common_letters_reward_func(parser, completion, **kwargs) -> float:
#     """Reward for guessing frequent letters"""
#     common_letters = "ETAOINSHRDLU"
#     messages = parser.get_user_messages(completion)[:5]  # First 5 guesses
#     common_guessed = 0
#     # Bonus for guessing E,T,A,O,I,N early
#     return 0.0


# def no_repeat_reward_func(parser, completion, **kwargs) -> float:
#     """Penalty for repeated guesses"""
#     messages = parser.get_user_messages(completion)
#     repeat_penalty = 0
#     # Check for "already been guessed" messages
#     return 0.0


# def word_length_aware_reward_func(parser, completion, answer, **kwargs) -> float:
#     """Scale rewards by word difficulty"""
#     base_reward = check_answer_reward_func(parser, completion, answer, **kwargs)
#     word_length = len(answer)
#     # Harder words (longer/rarer) get bonus multipliers
#     return 0.0

def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
):
    parser = vf.XMLParser(fields=["guess"], answer_field="guess")
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(check_answer_reward_func)

    vf_env = TextArenaEnv(
        game="Hangman-v0",
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
