from minicons import scorer
import argparse
from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config
from glob import glob
from tqdm import tqdm
import pandas as pd
import os


def read_data(data_path):
    test_set = {}
    phenomenon_paths = glob(f'{data_path}/*.tsv')
    for p in tqdm(phenomenon_paths):
        phenomenon = p.split('/')[-1].split('.')[0]
        sentences = pd.read_csv(p, sep='\t').to_dict(orient='records')
        if phenomenon=='yn_questions':
            sent_pair = [(x['sent1'], x['sent2']) for x in sentences]
            test_set[phenomenon] = sent_pair
        elif phenomenon=='wh_questions':
            sent_pair= [(x['sent1'], x['sent2'], x['sent3']) for x in sentences]
            test_set[phenomenon] = sent_pair
        else:
            raise ValueError(f'{phenomenon} is not supported yet!')
    return test_set

def eval_sent_pair(ilm_model, tokenizer, test_set):
    results = {}
    distributions = {}
    for phe, sents in tqdm(test_set.items()):
        correct = 0
        distribution = []
        s1_pref = 0
        s2_pref = 0
        s3_pref = 0
        for sent in sents:
            sent = list(sent)
            if len(sent)==2:
                num_token0 = len(tokenizer.encode(sent[0],add_special_tokens=False))
                num_token1 = len(tokenizer.encode(sent[1],add_special_tokens=False))
                nll0, nll1 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
                ppl0 = nll0 / num_token0
                ppl1 = nll1 / num_token1
                if ppl0 < ppl1:
                    s1_pref +=1
                else:
                    s2_pref +=1
            if len(sent)==3:
                num_token0 = len(tokenizer.encode(sent[0], add_special_tokens=False))
                num_token1 = len(tokenizer.encode(sent[1], add_special_tokens=False))
                num_token2 = len(tokenizer.encode(sent[2],add_special_tokens=False))

                nll0, nll1, nll2 = ilm_model.sequence_score(sent, reduction=lambda x: -x.sum(0).item())
                ppl0 = nll0 / num_token0
                ppl1 = nll1 / num_token1
                ppl2 = nll2 / num_token2
                if ppl0 < ppl1 and ppl0 < ppl2:
                    s1_pref += 1
                elif ppl1 < ppl0 and ppl1 < ppl2:
                    s2_pref += 1
                else:
                    s3_pref += 1
        s1_f = s1_pref / len(sents)
        s2_f = s2_pref / len(sents)
        s3_f = s3_pref / len(sents)
        acc = {'s1_pref': s1_f, 's2_pref': s2_f, 's3_pref': s3_f}
        results[phe] = acc
        print(phe, acc)
    return results, distributions

if __name__ == '__main__':
    args = argparse.ArgumentParser('eval language models')
    args.add_argument('model_name', type=str, help='model name')
    args = args.parse_args()
    os.makedirs(f'results', exist_ok=True)
    model_name = args.model_name
    best_checkpoint = args.best_checkpoint
    refs = list_repo_refs(model_name, repo_type="model")
    num_checkpoints = refs.branches
    checkpoints = sorted([x.name for x in num_checkpoints if 'main' not in x.name], key=lambda x: int(x.split('-')[-1]))
    test = read_data(f'eval_data')
    model_name_name = model_name.split('/')[-1]
    f_results = {}
    for checkpoint in checkpoints:
        results = {}
        print(model_name, checkpoint)
        ilm_model = scorer.IncrementalLMScorer(model_name, 'cpu',revision=checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        acc, dist = eval_sent_pair(ilm_model, tokenizer, test)
        results[checkpoint] = acc
        pd.DataFrame(results).to_csv(f'results/results_{model_name_name}_ckpt{checkpoint}.csv')