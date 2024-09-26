
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import datasets
import torch.nn.functional as F
import tqdm
import argparse
import json
from model import load_tokenizer, load_model
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics
import scipy
import math
import jsonlines
from bart_score import BARTScorer
import matplotlib.pyplot as plt
import time
pct = 0.015
def save_llr_histograms(prediction, name):
    plt.clf()
    plt.figure(figsize=(8,5))
    results = {'human-written':[],'LLM-generated':[]}
    for r in range(len(prediction['real'])):
        results['human-written'].append(prediction["real"][r])
        results['LLM-generated'].append(prediction["samples"][r])
    if name == "gpt2-xl":
        plt.hist([r for r in results['human-written']], alpha=0.5, bins=130, range=(0, 1.60), label='Human-written')
        plt.hist([r for r in results['LLM-generated']], alpha=0.5, bins=75, range=(0, 1.60), label='LLM-generated')
        plt.legend(loc='upper right',fontsize=21)
        plt.ylim(0, 100)
        plt.xticks(fontsize=20)
        plt.yticks([0, 20, 40, 60, 80, 100],fontsize=20)
    else:
        plt.hist([r for r in results['human-written']], alpha=0.5, bins=130, range=(0, 1.60))
        plt.hist([r for r in results['LLM-generated']], alpha=0.5, bins=75, range=(0, 1.60))
        plt.ylim(0, 100)
        plt.xticks(fontsize=20)
        plt.yticks([0, 20, 40, 60, 80, 100],fontsize=20)


    plt.savefig(f"sim_histograms_{name}.png")


def fill_and_mask(text,  pct = pct):
    tokens = text.split(' ')

    n_spans = pct * len(tokens)
    n_spans = int(n_spans)

    repeated_random_numbers = np.random.choice(range(len(tokens)), size=n_spans)

    return repeated_random_numbers.tolist()


def apply_extracted_fills(texts, indices_list=[]):
    tokens = [x.split(' ') for x in texts]

    for idx, (text, indices) in enumerate(zip(tokens, indices_list)):
        for idx in indices:
            text[idx] = ""

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, pct=pct):

    indices_list = [fill_and_mask(x, pct) for x in texts]
    perturbed_texts = apply_extracted_fills(texts, indices_list)

    return perturbed_texts


def perturb_texts(texts, pct=pct):
    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), 50), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + 50], pct))
    return outputs

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples_2 = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples_2

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_score(logits_ref, logits_score, labels,source_texts, perturbed_texts, basemodel):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples_2 = get_samples(logits_ref, labels)

    log_likelihood_x = get_likelihood(logits_score, labels)
    log_rank_x = get_logrank(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples_2)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

    source_texts_list = [source_texts]*10

    #bart-score
    values = bart_scorer.score(perturbed_texts,source_texts_list, batch_size=10)
    mean_values = np.mean(values)
    
    if basemodel == 'Fast':
        if 'gemini' in args.dataset_file and 'pubmed' in args.dataset_file:
            #Fast-DetectGPT has too many negative values in the output scores on the data generated by Gemini on PubMed, which can lead to scaling of the improvement effect. A constant is added here to address this.
            output_score = (((log_likelihood_x.squeeze(-1).item()  - miu_tilde.item())/ (sigma_tilde.item()))+2) * math.pow(math.e, -mean_values)
        else:
            output_score = ((log_likelihood_x.squeeze(-1).item() - miu_tilde.item()) / (sigma_tilde.item())) * math.pow(math.e, -mean_values)
    elif basemodel == 'lrr':
        output_score = (log_likelihood_x.squeeze(-1).item() / log_rank_x) * math.pow(math.e, -mean_values)
    elif basemodel == 'likelihood':
        output_score = log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)
    elif basemodel == 'logrank':
        output_score = log_rank_x * math.pow(math.e, mean_values)
    elif basemodel == 'standalone':
        output_score = -mean_values
    #The default base model is set to likelihood
    else:
        output_score =  log_likelihood_x.squeeze(-1).item() * math.pow(math.e, mean_values)

    return output_score

def experiment():

    # load model
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset,args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()

    data = load_data(args.dataset_file)
    n_originals = len(data['original'])
    n_samples = len(data["sampled"])
    start_time = time.time()
    # evaluate criterion

    name = args.basemodel
    criterion_fn = get_score

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []

    perturbed_original_texts = perturb_texts([x for x in data['original'] for _ in range(10)])
    perturbed_sampled_texts = perturb_texts([x for x in data['sampled'] for _ in range(10)])

    perturbed_original_texts_list = []
    perturbed_sampled_texts_list = []

    for idx in range(len(data['original'])):
        perturbed_original_texts_list.append(perturbed_original_texts[idx * 10: (idx + 1) * 10])
        perturbed_sampled_texts_list.append(perturbed_sampled_texts[idx * 10: (idx + 1) * 10])


    for idx in tqdm.tqdm(range(n_originals),desc="computing"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():

            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            
            original_crit = criterion_fn(logits_ref, logits_score, labels, original_text,perturbed_original_texts_list[idx], args.basemodel)

        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, sampled_text, perturbed_sampled_texts_list[idx], args.basemodel)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results
    results_file = f'{args.output_file}.{name}.json'
    results = {'name': f'{name}_threshold',
               'info': {'n_samples': n_samples},
               'predictions': predictions,
               'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
               'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

    #save_llr_histograms(predictions,args.scoring_model_name)
    end_time = time.time()
    print(f"cost of time：{args.scoring_model_name}",end_time - start_time)
    return roc_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./results_/")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="data_test/")
    parser.add_argument('--reference_model_name', type=str, default="gpt2")
    parser.add_argument('--scoring_model_name', type=str, default="gpt2")
    parser.add_argument('--basemodel', type=str, default="Fast")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    base_tokenizer = scoring_tokenizer

    DEVICE = "cuda:0"
    bart_scorer = BARTScorer(device=DEVICE, checkpoint='facebook/bart-base')

    experiment()

