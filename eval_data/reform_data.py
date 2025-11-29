from pathlib import Path
sents = Path('yesno_question.txt').read_text().strip().split('\n')

with open('yn_questions.tsv', 'w') as f:
    f.write('sent1\tsent2\n')
    for i, sent in enumerate(sents):
        if i%2==0:
            sent1 = sent
            sent2 = sents[i+1]
            f.write(f'{sent1}\t{sent2}\n')

