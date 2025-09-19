import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv

def check_answer_reward_func(parser, completion, answer, **kwargs) -> float:
    guess = parser.parse_answer(completion)
    return 1.0 if guess == "[" + answer.upper() + "]" else 0.0

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