from collections import Counter
import os
from datetime import datetime
import ujson
import pickle
import sys
from sklearn.metrics.pairwise import cosine_similarity
import regex as re

from tqdm import tqdm
from nltk import word_tokenize
from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
import argparse
import numpy as np
import pandas as pd
from transformers import pipeline
np.random.seed(1992)
import json
import torch
from evaluate import load
import stanza
from nltk.corpus import stopwords
import string
from eval.model_loader import add_args, load_model_and_apply_patches

SAP_BERT = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
META_COLS = ['plan_recall', 'plan_precision', 'plan_f1']
IN_DIR = '/nlp/projects/summarization/bhc_data_cleanup'
SPAN_EMBED_DIM = 768
_DEFAULT_PRED_ENT_THRESHOLD = 0.75
_DEFAULT_ENT_MERGE_THRESHOLD = 0.6
BHC_PREFIX = '\n\n### PARTIAL HOSPITAL COURSE:\n'
BHC_FULL = '\n\n### BRIEF HOSPITAL COURSE:\n'


PATIENT_TERMS = {'patient', 'pt', 'patient\'s', 'patients', 'patients\''}
BHC_STOPWORDS = set(stopwords.words('english')).union(string.punctuation).union(PATIENT_TERMS)


def extract_generated_ents(pred_sents, tools, ent_merge_threshold=_DEFAULT_ENT_MERGE_THRESHOLD):
    pred_ents = extract_target_ents(pred_sents, tools['nlp'])
    pred_ent_types = Counter([x['type'] for x in pred_ents])

    pred_ent_spans = [x['text'] for x in pred_ents]
    if len(pred_ents) == 0:
        pred_ent_embeds = []
    else:
        pred_ent_embeds = embed_concept_spans(
            tools['sapbert_model'], tools['sapbert_tokenizer'], pred_ent_spans
        )

    if len(pred_ent_embeds) > 0:
        pred_self_sim = cosine_similarity(pred_ent_embeds, pred_ent_embeds)
        V = len(pred_self_sim)
        self_synonym_matrix = pred_self_sim >= ent_merge_threshold
        pred_graph = Graph(V)
        for i in range(V):
            for j in range(i + 1, V):
                if self_synonym_matrix[i, j]:
                    pred_graph.addEdge(i, j)

        pred_component_idxs = pred_graph.connectedComponents()
    else:
        pred_component_idxs = []
    pred_ent_clusters = [[pred_ent_spans[i] for i in c] for c in pred_component_idxs]

    pred_ent_cluster_types = [
        pred_ents[pred_ent_spans.index(c[0])]['type'] for c in pred_ent_clusters
    ]

    return {
        'embeds': pred_ent_embeds,
        'ents': pred_ents,
        'ent_types': pred_ent_types,
        'spans': pred_ent_spans,
        'cluster_idxs': pred_component_idxs,
        'cluster_spans': pred_ent_clusters,
        'cluster_types': pred_ent_cluster_types,
    }


def gen_single_sentence(pipe, prompt):
    n = len(prompt)
    response = pipe(
        prompt, num_return_sequences=1, max_new_tokens=args.max_new_tokens,
    )[0]["generated_text"][n:]

    # TODO change to end on newline
    return response.strip().split('\n')[0]


def process_prediction(args, text, tools):
    sents = [x.strip() for x in text.split('\n') if len(x.strip()) > 0]
    out_obj = extract_generated_ents(sents, tools, ent_merge_threshold=args.ent_merge_threshold)
    out_obj['pred_sents'] = sents

    gen_span_to_embed = {
        a: b for a, b in zip(out_obj['spans'], out_obj['embeds'])
    }

    out_obj['gen_span_to_embed'] = gen_span_to_embed

    gen_ent_cluster_embeds = [
        np.array([gen_span_to_embed[k] for k in cluster]) for cluster in out_obj['cluster_spans']
    ]

    out_obj['gen_ent_cluster_embeds'] = gen_ent_cluster_embeds

    return out_obj


def remove_dup_lines(text):
    arr = text.split('\n')
    new_arr = []
    for x in arr:
        x = x.strip()
        if x not in new_arr and len(x) > 0:
            new_arr.append(x)
    return '\n'.join(new_arr)


# https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
class Graph:
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def DFSUtil(self, temp, v, visited):

        # Mark the current vertex as visited
        visited[v] = True

        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc


def remove_non_ents(text, remove_title=True):
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if 'title:' in line.lower() and not remove_title:
            new_lines.append(line)
        elif '<doc-sep>' in line.lower() and not remove_title:
            new_lines.append(line)
        elif line == '':
            new_lines.append(line)
        elif '{{' in line or '<e>' in line:
            new_lines.append(line)
    return '\n'.join(new_lines)


def remove_non_ents_from_list(text, rel_spans, remove_title=True):
    rel_spans_lower = [x.strip(string.punctuation).lower() for x in list(rel_spans)]
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        if 'title:' in line.lower() or '<doc-sep>' in line.lower():
            if not remove_title:
                new_lines.append(line)
        elif line == '':
            new_lines.append(line)
        elif any([x_lower in line.lower() for x_lower in rel_spans_lower]):
            new_lines.append(line)
    return '\n'.join(new_lines)


def sents_from_html(html_str):
    tps = html_str.split('<SEP>')
    return [tps[idx + 1] for idx, tp in enumerate(tps) if tp.startswith('<s') and idx + 1 < len(tps)]


def split_into_notes(html_str):
    tps = html_str.split('<SEP>')
    notes = []
    curr_note = []
    for tp in tps:
        curr_note.append(tp)
        if tp == '</d>':
            notes.append('<SEP>'.join(curr_note))
            curr_note = []
    return notes


def take_latest(source_transform, max_toks=8192):
    curr_num_toks = 0
    lines = source_transform.split('\n')
    trunc = []
    for end_idx in range(len(lines) - 1, -1, -1):
        line = lines[end_idx]
        curr_num_toks += len(line.split(' '))
        trunc.insert(0, line)
        if curr_num_toks > max_toks:
            break
    source_transform = '\n'.join(trunc)
    return source_transform


def remove_duplicates_preserve_order(arr):
    """
    Removes duplicates from a list while preserving order.

    :param arr: A list with possible duplicates.
    :return: A new list with duplicates removed, in the same order as the original list.
    """
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_attr(tag, attr):
    return re.search(r'\s' + attr + r'=([^ ]+)', tag).group(1).strip('<>: ')


def embed_concept_spans(model, tokenizer, syns, batch_size=4096, verbose=False):
    all_reps = []
    batch_starts = np.arange(0, len(syns), batch_size)
    num_batches = len(batch_starts)
    batch_ct = 0
    for i in batch_starts:
        batch_ct += 1
        toks = tokenizer.batch_encode_plus(
            syns[i:i + batch_size], padding='max_length', max_length=25, truncation=True, return_tensors='pt')
        toks_cuda = {k: v.to(model.device) for k, v in toks.items()}
        with torch.no_grad():
            output = model(**toks_cuda)
            cls_rep = output[0][:, 0, :]
            all_reps.append(cls_rep.cpu().detach().numpy())
        if verbose:
            print(f'{batch_ct}/{num_batches}')
    return np.concatenate(all_reps, axis=0)


def transform_text_for_llama(
        html_str, include_header=True, include_title=True, include_sent_markers=False, meta=None
):
    tps = html_str.split('<SEP>')
    curr_str = ''
    note_idx = -1
    for idx, tp in enumerate(tps):
        if tp.startswith('<d') and include_title:
            if idx > 0 and tps[idx - 1].startswith('<d'):
                continue
            try:
                curr_note_id = get_attr(tp, 'note_id')
                note_title = ' '.join(map(str.capitalize, curr_note_id.split('-')[9:]))
            except:
                title = get_attr(tp, 'title')
                note_title = ' '.join(map(str.capitalize, title.split('_')))
            note_idx += 1
            curr_str += '\n\n### Title: ' + note_title + '\n'
            if meta is not None:
                note_meta = meta[note_idx]
                curr_str += note_meta + '\n'
        elif tp.startswith('<h'):
            raw = get_attr(tp, 'raw')
            if raw.lower() == 'unknown':
                continue
            raw_section = re.sub(r'[_\s]+', ' ', raw).strip()
            if len(curr_str) > 0 and curr_str[-1] != '\n':
                curr_str += '\n'
            if include_header:
                curr_str += raw_section + ':\n'
        elif idx > 0 and tps[idx - 1].startswith('<s'):
            # sent_str = remove_tags_from_sent(tp)
            if len(curr_str) > 0 and curr_str[-1] not in {'\n', '\t', ' '}:
                curr_str += ' '
            if include_sent_markers:
                curr_str += '<s>' + tp + '</s>'
            else:
                curr_str += tp
    return curr_str.strip()


def get_entity_guidance(
        example_id, all_ent_probs, source_ent_clusters, source_ent_types, max_csize=15,
        pred_ent_threshold=_DEFAULT_PRED_ENT_THRESHOLD, min_ents=3, max_ents=100
):
    # priority = np.argsort(-np.array(ent_probs))
    ent_probs = [x for x in all_ent_probs if x['example_id'] == example_id][0]

    pred_idxs = get_pred_ent_cluster_idxs(ent_probs['cluster_pred_probs'], pred_ent_threshold, min_ents, max_ents)

    pred_source_clusters = [source_ent_clusters[i] for i in pred_idxs]
    pred_source_types = [source_ent_types[i] for i in pred_idxs]
    target_problems = [c for c, t in zip(pred_source_clusters, pred_source_types) if t == 'PROBLEM']
    target_tests = [c for c, t in zip(pred_source_clusters, pred_source_types) if t == 'TEST']
    target_treatments = [c for c, t in zip(pred_source_clusters, pred_source_types) if t == 'TREATMENT']

    ents = {
        'problems': target_problems,
        'treatments': target_treatments,
        'tests': target_tests,
    }

    problems = '\n'.join(['; '.join(z[:min(len(z), max_csize)]) for z in target_problems])
    treatments = '\n'.join(['; '.join(z[:min(len(z), max_csize)]) for z in target_treatments])
    tests = '\n'.join(['; '.join(z[:min(len(z), max_csize)]) for z in target_tests])
    guidance = f'# PROBLEMS:\n{problems}\n\n# TREATMENTS:\n{treatments}\n\n# TESTS:\n{tests}'
    return guidance, ents, pred_source_clusters


def load_ent_embeds():
    print('Loading embeddings...')
    with open('/nlp/projects/summarization/bhc_data_cleanup/entity_embeddings_test.pk', 'rb') as fd:
        span2embed = pickle.load(fd)
    return span2embed


def load_ent_info(args, example_id, span2embed):
    ent_suffix = '' if args.dataset == 'epic' else f'_{args.dataset}'
    entity_fn = os.path.join(IN_DIR, f'entity_stanza{ent_suffix}', f'{example_id}.json')
    entity_merge_fn = os.path.join(IN_DIR, f'entity_stanza{ent_suffix}_top_ents', f'{example_id}.json')

    with open(entity_merge_fn, 'r') as fd:
        ent_merges = ujson.load(fd)

    source_ent_clusters = ent_merges['source_cluster_spans']
    source_ent_types = ent_merges['source_cluster_types']

    with open(entity_fn, 'r') as fd:
        ents = ujson.load(fd)
        target_ents = ents['target']

        target_span_cts = Counter([x['text'] for x in target_ents])
        target_spans = list(sorted(list(target_span_cts.keys())))
        target_embeds = np.array([span2embed[span] for span in target_spans])

        source_ents = ents['source']
        source_span_cts = Counter([x['text'] for x in source_ents])
        source_spans = list(sorted(list(source_span_cts.keys())))
        source_embeds = np.array([span2embed[span] for span in source_spans])

    with open(entity_merge_fn, 'r') as fd:
        merges = ujson.load(fd)

    return {
        'ent_merges': merges,
        'source_ent_clusters': source_ent_clusters,
        'source_ents': source_ents,
        'source_ent_types': source_ent_types,
        'source_embeds': source_embeds,
        'target_embeds': target_embeds,
    }


def generate_note_meta(note_tag, note_idx, num_notes, admit_date, discharge_date, now=None):
    if now is None:
        now = datetime.strptime(get_attr(note_tag, 'time').split('_')[0], "%m-%d-%y").date()
    days_into_stay = (now - admit_date).days
    los = (discharge_date - admit_date).days
    is_day_of_admission = now == admit_date
    is_day_of_discharge = now == discharge_date
    now_str = now.strftime("%m/%d/%Y")
    meta = f"DATE: {now_str}\n\nNOTE ORDER: {note_idx + 1} of {num_notes}\n\nDAY: {days_into_stay} of {los}"

    if is_day_of_admission:
        meta += "\n\nON DAY OF ADMISSION"
    if is_day_of_discharge:
        meta += "\n\nON DAY OF DISCHARGE"
    return meta


def remove_duplicates(arr):
    seen = set()
    covered_toks = set()
    new_arr = []
    for sent in arr:
        sent_toks = list(set([x.strip().lower() for x in re.split('\W+', sent) if x.strip().lower() not in BHC_STOPWORDS and len(x.strip()) > 0]))
        num_new = len([
            tok for tok in sent_toks if tok not in covered_toks
        ])
        if num_new < 2 or sent.lower() in seen:
            continue
        for tok in sent_toks:
            covered_toks.add(tok)
        new_arr.append(sent)
        seen.add(sent.lower())
    return arr


def extract_target_ents(target_sents, nlp):
    concepts = []
    for target_idx, sent in enumerate(target_sents):
        ents = nlp(sent).entities
        ents = [{'text': ent.text, 'type': ent.type} for ent in ents]
        for ent in ents:
            ent.update({'sent_idx': target_idx})
            concepts.append(ent)
    return concepts


def get_pred_ent_cluster_idxs(cluster_pred_probs, pred_ent_threshold, min_ents, max_ents):
    priority = np.argsort(-np.array(cluster_pred_probs)).tolist()
    probs_sorted = [cluster_pred_probs[j] for j in priority]

    pred_idxs = [idx for idx, score in zip(priority, probs_sorted) if score >= pred_ent_threshold]

    if len(pred_idxs) < min_ents:
        pred_idxs = priority[:min_ents]
    elif len(pred_idxs) > max_ents:
        pred_idxs = priority[:max_ents]
    return pred_idxs


def extract_pred_ent_span_set(
        ent_probs, ent_info, pred_ent_threshold=_DEFAULT_PRED_ENT_THRESHOLD, min_ents=3, max_ents=100
):
    pred_idxs = get_pred_ent_cluster_idxs(ent_probs['cluster_pred_probs'], pred_ent_threshold, min_ents, max_ents)

    pred_source_ent_set = set()
    for x in [ent_info['source_ent_clusters'][i] for i in pred_idxs]:
        for y in x:
            pred_source_ent_set.add(y)
    return pred_source_ent_set


def generate_input(notes, admit_date=None, discharge_date=None):
    num_notes = len(notes)
    outputs = []
    for note_idx in range(len(notes)):
        note = notes[note_idx]
        tags = note.split('<SEP>')
        if tags[0] == tags[1]:
            note = '<SEP>'.join(tags[1:])

        if admit_date is None:
            note_str = transform_text_for_llama(
                note, include_header=True, include_title=True, include_sent_markers=False
            )
        else:
            meta = generate_note_meta(
                tags[0], note_idx, num_notes, admit_date=admit_date, discharge_date=discharge_date
            )
            note_str = transform_text_for_llama(
                note, include_header=True, include_title=True, include_sent_markers=False, meta=[meta]
            )
        note_str = note_str.replace('?', ' ').replace(u'\xa0', ' ')
        outputs.append(note_str)
    return '\n\n'.join(outputs)


def filter_to_max_token_limit(source_input, clusters, max_prompt_tokens):
    lines = [x.strip() for x in source_input.split('\n') if len(x.strip()) > 0]
    line_cluster_idxs = []
    line_toks = []
    line_titles = []
    curr_title = ''
    for line in lines:
        if line.lower().startswith('### title'):
            curr_title = line
        covered_idxs = []
        line_lower = line.lower()
        for cluster_idx, cluster in enumerate(clusters):
            is_covered = False
            for ent in cluster:
                if ent.lower() in line_lower:
                    is_covered = True
                    break
            if is_covered:
                covered_idxs.append(cluster_idx)
        line_cluster_idxs.append(covered_idxs)
        line_toks.append(len(word_tokenize(line)))
        line_titles.append(curr_title)

    clusters_accounted_for = set()
    chosen_toks = 0
    chosen_idxs = []

    for _ in range(len(lines)):
        priorities = [
            len(set(idxs) - clusters_accounted_for) for idxs in line_cluster_idxs
        ]
        chosen_idx = np.argmax(priorities)
        max_priority = max(priorities)
        if max_priority == 0:
            break
        chosen_idxs.append(chosen_idx)
        chosen_toks += line_toks[chosen_idx]
        for to_add in line_cluster_idxs[chosen_idx]:
            clusters_accounted_for.add(to_add)

        if chosen_toks >= max_prompt_tokens:
            break

    chosen_idxs = list(sorted(chosen_idxs))
    if chosen_toks < max_prompt_tokens:
        # Do the same thing without the coverage (just pick sentences with already covered until max token limit)
        for _ in range(len(lines)):
            priorities = [len(set(idxs)) for idxs in line_cluster_idxs]
            for chosen_idx in chosen_idxs:
                priorities[chosen_idx] = 0
            chosen_idx = np.argmax(priorities)
            max_priority = max(priorities)
            if max_priority == 0:
                break
            chosen_idxs.append(chosen_idx)
            chosen_toks += line_toks[chosen_idx]
            for to_add in line_cluster_idxs[chosen_idx]:
                clusters_accounted_for.add(to_add)

            if chosen_toks >= max_prompt_tokens:
                break

    chosen_idxs = list(sorted(chosen_idxs))
    assert len(chosen_idxs) == len(set(chosen_idxs))
    chosen_titles = [line_titles[i] for i in chosen_idxs]
    chosen_lines = [lines[i] for i in chosen_idxs]

    output_lines = []
    curr_title = ''
    for title, line in zip(chosen_titles, chosen_lines):
        if title != curr_title:
            output_lines.append('\n')
            output_lines.append(title)
            curr_title = title
            output_lines.append('\n')
        output_lines.append(line)

    unaccounted_for_clusters = [cluster for idx, cluster in enumerate(clusters) if idx not in clusters_accounted_for]
    return '\n'.join(output_lines).strip(), unaccounted_for_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BHC Partial Summarization.')
    parser.add_argument('--data_dir', default='/nlp/projects/summarization/bhc_data_cleanup')

    parser.add_argument('--dataset', default='epic')
    parser.add_argument('--experiment', default='sent_frost_unlike')

    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-overwrite', default=False, action='store_true')

    parser.add_argument('--max_examples', default=1000, type=int)
    parser.add_argument('--max_gen_sents', default=50, type=int)
    parser.add_argument('--ent_merge_threshold', default=0.6, type=float)
    parser.add_argument('--max_summary_tokens', default=512, type=int)
    parser.add_argument('--max_prompt_tokens', default=4096, type=int)
    parser.add_argument(
        '--stagant_breaking_pt', default=5, type=int,
        help='Number of times we can generate a sentence that does not cover any new entities before breaking.'
    )

    # Clique Parameter
    parser.add_argument('--split', default='test')
    parser.add_argument('-human', default=False, action='store_true')
    parser.add_argument('--pred_ent_threshold', default=0.81, type=float)

    # Llama Arguments
    parser.add_argument('--max_new_tokens', type=int, default=128)
    # parser.add_argument('--repetition-penalty', type=float, default=1.18)
    parser.add_argument('--ckpt', default='latest')

    args = add_args(parser).parse_args()

    print('Loading SAPBERT')
    sapbert_tokenizer = AutoTokenizer.from_pretrained(SAP_BERT)
    sapbert_model = AutoModel.from_pretrained(SAP_BERT).eval().to(args.device)
    nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, use_gpu=True)
    rouge = load('rouge', keep_in_memory=True)

    tools = {
        'rouge': rouge,
        'nlp': nlp,
        'sapbert_model': sapbert_model,
        'sapbert_tokenizer': sapbert_tokenizer,
    }
    # End of loading tools

    # Model loading part
    args.model = f'/nlp/projects/summarization/bhc_data_cleanup/bhc_weights/yarn-7b-8k-{args.experiment}'
    print(f'Loading model from {args.model}')
    args.yarn = 16.0
    args.finetuned = True
    args.original_max_position_embeddings = 4096
    args.flash_attention = True
    args.custom_model_together = True

    tokenizer_model = 'NousResearch/Llama-2-7b-hf'  # Switched from args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, model_max_length=sys.maxsize, trust_remote_code=True)

    # tokenizer.eos_token_id = tokenizer.encode('\n')[-1]
    # tokenizer.eos_token = '\n'

    if args.ckpt == 'final':
        model_path = args.model
    else:
        # Use latest checkpoint
        ckpts = [x for x in os.listdir(args.model) if 'step' in x]
        print(f'Checkpoints:\n', '\n'.join(ckpts))
        if args.ckpt == 'latest':
            steps = [int(x.split('_')[1]) for x in ckpts]
            ckpt_idx = int(np.argmax(steps))
        else:
            ckpt_idx = int([x.split('_')[1] for x in ckpts].index(args.ckpt))

        model_path = os.path.join(args.model, ckpts[ckpt_idx])
    print(f'Loading from {model_path}...')
    model = load_model_and_apply_patches(model_path, args)
    print(f'Loaded from {model_path}!')

    pipe_kwargs = {
        'do_sample': False,
        # 'no_repeat_ngram_size': 3,
        # 'repetition_penalty': args.repetition_penalty
    }
    print('Creating Transformers pipeline to serve predictions with following hparams...')
    for k, v in pipe_kwargs.items():
        print(f'{k} -> {v}')
    pipe = pipeline(
        'text-generation', model=model, tokenizer=tokenizer,  # pad_token_id=tokenizer.eos_token_id,
        **pipe_kwargs
    )
    print('Created pipeline!')

    out_dir = os.path.join(model_path, 'predictions')
    out_fn = f'{out_dir}.csv'
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == 'epic':
        if args.human:
            ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_human.json')
        else:
            ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_test.json')
    elif args.dataset == 'cumc':
        ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_cumc_test.json')
    else:
        assert args.dataset == 'mimic'
        ent_fn = os.path.join(IN_DIR, 'bhc_weights', 'fixed', 'test_mimic_test.json')

    with open(ent_fn, 'r') as fd:
        all_ent_probs = ujson.load(fd)

    print('Reading in dataset...')
    visit_meta = {}
    if args.dataset == 'cumc':
        data_dir = f'/nlp/projects/summarization/bhc_data_cleanup/cumc_test'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)
    elif args.dataset == 'epic':
        data_dir = '/nlp/projects/summarization/bhc_data_cleanup/summarization_deduped_dataset'
        visit_meta = pd.read_csv('/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv')
        visit_meta = {
            row['visit_id']: row for row in visit_meta.to_dict('records')
        }
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)[args.split]
    else:
        data_dir = '/nlp/projects/summarization/bhc_data_cleanup/mimic_test_filt'
        print(f'Reading in data from {data_dir}')
        data = load_from_disk(data_dir)

    if args.dataset == 'epic' and args.split == 'test':
        if args.human:
            valid_visit_ids = set(map(str, pd.read_csv(
                '/nlp/projects/summarization/bhc_data_cleanup/bhc_human_meta.csv'
            )['visit_id']))
        else:
            valid_visit_ids = set(map(str, pd.read_csv(
                '/nlp/projects/summarization/bhc_data_cleanup/bhc_test_meta.csv'
            )['visit_id']))
        data = data.filter(
            lambda row: row['visit_id'] in valid_visit_ids
        )
        print(f'{len(data)} examples after removing contaminated samples.')

    n = len(data)
    if args.max_examples is not None and args.max_examples < n:
        idxs = list(sorted(np.random.choice(np.arange(n), size=(args.max_examples), replace=False)))
        data = data.select(idxs)

    example_ids = set([x['example_id'] for x in all_ent_probs])
    prev = len(data)
    data = data.filter(lambda row: row['example_id'] in example_ids)
    new = len(data)
    print(f'Entity Probabilities for {new} / {prev} examples. Filtering...')

    span2embed = load_ent_embeds()

    scores = []
    outputs = []

    num_stagnant_ctr = 0

    for example in tqdm(data):
        example_id = example['example_id']
        save_fn = os.path.join(out_dir, f'{example_id}.json')

        if os.path.exists(save_fn) and not args.overwrite:
            print(f'Already exists --> {save_fn}. Skipping...')
            with open(save_fn, 'r') as fd:
                out_row = ujson.load(fd)
                scores.append(out_row.copy())
                outputs.append(out_row.copy())
            continue

        # Entity Stuff
        ent_probs = [x for x in all_ent_probs if x['example_id'] == example_id][0]
        ent_info = load_ent_info(args, example_id, span2embed)
        guidance, ents_in_guidance, pred_source_clusters = get_entity_guidance(
            example_id, all_ent_probs, ent_info['source_ent_clusters'], ent_info['source_ent_types'],
            pred_ent_threshold=args.pred_ent_threshold
        )

        target_no_dup = '\n'.join(remove_duplicates_preserve_order(example['target_sents']))

        pred_source_ent_set = extract_pred_ent_span_set(
            ent_probs, ent_info, pred_ent_threshold=args.pred_ent_threshold
        )

        notes = split_into_notes(example['source'])

        cluster_is_covered = {
            json.dumps(c): False for c in pred_source_clusters
        }

        cluster_embeds = {
            json.dumps(c): embed_concept_spans(tools['sapbert_model'], tools['sapbert_tokenizer'], c)
            for c in pred_source_clusters
        }

        pred_sents = []
        num_tokens_generated = 0

        for step in range(args.max_gen_sents):
            uncovered_by_type = {
                k: [z for z in v if not cluster_is_covered[json.dumps(z)]] for k, v in ents_in_guidance.items()
            }

            uncovered_clusters = uncovered_by_type['problems'] + uncovered_by_type['treatments'] + uncovered_by_type['tests']

            uncovered_source_set = set()
            for k, arr in uncovered_by_type.items():
                for cluster in arr:
                    for ent in cluster:
                        uncovered_source_set.add(ent)

            admit_date = discharge_date = None
            if 'first_date' in example:
                admit_date = datetime.strptime(example['first_date'].split('_')[0], "%m-%d-%y").date()
                discharge_date = datetime.strptime(example['last_date'].split('_')[0], "%m-%d-%y").date()

            source_input = generate_input(notes, admit_date=admit_date, discharge_date=discharge_date)

            source_input, unaccounted_for_clusters = filter_to_max_token_limit(
                source_input, uncovered_clusters, args.max_prompt_tokens
            )
            unaccounted_for_clusters_json = list(map(ujson.dumps, unaccounted_for_clusters))
            source_input = re.sub(r'\n{2,}', '\n\n', source_input).strip()

            # We might have filtered out some of the text when truncating to maximum window
            uncovered_by_type_accounted_for = {
                'problems': [
                    c for c in uncovered_by_type['problems'] if ujson.dumps(c) not in unaccounted_for_clusters_json
                ],
                'treatments': [
                    c for c in uncovered_by_type['treatments'] if ujson.dumps(c) not in unaccounted_for_clusters_json
                ],
                'tests': [
                    c for c in uncovered_by_type['tests'] if ujson.dumps(c) not in unaccounted_for_clusters_json
                ],
            }

            uncovered_problem_str = '; '.join([x[0] for x in uncovered_by_type_accounted_for['problems']])
            uncovered_treatment_str = '; '.join([x[0] for x in uncovered_by_type_accounted_for['treatments']])
            uncovered_test_str = '; '.join([x[0] for x in uncovered_by_type_accounted_for['tests']])

            guidance = f'### PROBLEMS: {uncovered_problem_str} ### TREATMENTS: ' \
                       f'{uncovered_treatment_str} ### TESTS: {uncovered_test_str}'

            if len(pred_sents) == 0:
                pred_str = ''
            else:
                pred_str = '\n'.join(pred_sents) + '\n\n'

            prompt = f'{source_input}\n\n### BRIEF HOSPITAL COURSE:\n\n{pred_str}{guidance}\n### SENTENCE: '

            pred_sent = gen_single_sentence(pipe=pipe, prompt=prompt)
            sent_obj = process_prediction(args, pred_sent, tools)
            if pred_sent in pred_sents:
                print('Predicted already generated sentence. Breaking and returning summary.')
                break
            pred_sents.append(pred_sent)
            num_tokens_generated += len(word_tokenize(pred_sent))

            print('Current Prediction...')
            print('\n'.join(pred_sents))
            print('\n')

            num_gen_ents = len(sent_obj['ents'])

            prev_coverage_ct = sum(list(cluster_is_covered.values()))

            if num_gen_ents > 0:
                for cluster_key, embeds in cluster_embeds.items():
                    if not cluster_is_covered[cluster_key]:
                        is_aligned = np.any(cosine_similarity(sent_obj['embeds'], embeds) >= args.ent_merge_threshold)
                        cluster_is_covered[cluster_key] = is_aligned

            coverage_ct = sum(list(cluster_is_covered.values()))
            print(f'Coverage: {coverage_ct}/{len(cluster_is_covered)}')

            gain = coverage_ct - prev_coverage_ct

            if gain == 0:
                num_stagnant_ctr += 1
            else:
                # Reset the counter
                num_stagnant_ctr = 0

            if num_stagnant_ctr >= args.stagant_breaking_pt:
                print(f'Generated {num_stagnant_ctr} in a row. Breaking.')
                break

            if coverage_ct == len(cluster_is_covered):
                print(f'Covered all entities! Breaking...')
                break

            # Check if all entities are covered
            if num_tokens_generated >= args.max_summary_tokens:
                print(f'Surpassed max token limit: {num_tokens_generated}/{args.max_summary_tokens} used. Breaking.')
                break

        prediction = '\n'.join(pred_sents)

        print('\n\n')
        print(prediction)
        print('\n\n')

        out_row = {
            'example_id': example['example_id'],
            'reference': target_no_dup,
            'prediction': prediction
        }

        v_meta = visit_meta.get(example.get('visit_id', ''), {})
        out_row.update(v_meta)

        print(f'Saving to {save_fn}')
        with open(save_fn, 'w') as fd:
            json.dump(out_row, fd)

        outputs.append(out_row)

    outputs = pd.DataFrame(outputs)

    print(f'Saving predictions to {out_fn}...')
    outputs.to_csv(out_fn, index=False)

    print(outputs.select_dtypes(include='number').mean())
