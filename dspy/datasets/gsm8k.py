import random

import tqdm
from datasets import load_dataset


class GSM8K:
    def __init__(self) -> None:
        super().__init__()
        self.do_shuffle = False

        dataset = load_dataset("gsm8k", 'main')

        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            question = example['question']

            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))

            official_train.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            question = example['question']

            answer = example['answer'].strip().split()
            assert answer[-2] == '####'
            
            gold_reasoning = ' '.join(answer[:-2])
            answer = str(int(answer[-1].replace(',', '')))

            official_test.append(dict(question=question, gold_reasoning=gold_reasoning, answer=answer))

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]

        import dspy

        trainset = [dspy.Example(**x).with_inputs('question') for x in trainset]
        devset = [dspy.Example(**x).with_inputs('question') for x in devset]
        testset = [dspy.Example(**x).with_inputs('question') for x in testset]

        # print(f"Trainset size: {len(trainset)}")
        # print(f"Devset size: {len(devset)}")
        # print(f"Testset size: {len(testset)}")

        self.train = trainset
        self.dev = devset
        self.test = testset



def parse_integer_answer(answer, only_first_line=True):
    try:
        if only_first_line:
            answer = answer.strip().split('\n', 1)[0]
        
        # Efficiently find the last token containing a digit using reversed and next (avoids list creation)
        tokens = answer.split()
        for token in reversed(tokens):
            for c in token:
                if c.isdigit():
                    # Use partition to avoid split + index
                    int_token = token.partition('.')[0]
                    # More efficient digit extraction using generator expression
                    digits = ''.join(c for c in int_token if c.isdigit())
                    answer = int(digits)
                    return answer
        # If loop finishes, no token contains a digit
        answer = 0
    except (ValueError, IndexError):
        # print(answer)
        answer = 0

    return answer


def gsm8k_metric(gold, pred, trace=None):
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
