import math
import torch
import scipy.stats as stats
import ast
def calc_indices(args):
    #calc fold indices
    indices = []
    numfolds = args.xfolds
    len_dataset = 4330
    fold_size = math.ceil(len_dataset/numfolds)
    for fold in range(numfolds):
        startidx = fold*fold_size
        endidx = startidx+fold_size if startidx+fold_size < len_dataset else len_dataset
        indices.append((startidx, endidx))
    return indices

def read_in_labels(labels_file):
    labels = []
    inv_labels = {}
    with open(labels_file) as l:
        for line in l:
            line = line.strip().split("\t")
            labels.append('_'.join(line[1].split(' ')))
            inv_labels[line[1]] = int(line[0])
    return labels, inv_labels

def read_in_dialogues(dialogue_file):
    full_dials = {}
    dialogue_index = -1
    turn_index = -1
    with open(dialogue_file) as l:
        for line in l:
            if line.startswith('#S'):
                dialogue_index += 1
                turn_index = 0
            else:
                turn = line.strip().split('\t')
                full_dials[(dialogue_index, turn_index)] = turn
                turn_index += 1
    return full_dials

def read_in_dial_turn_idxs(dialogue_file):
    dialogue_indices = []
    dialogue_index = -1
    turn_index = -1
    assert(dialogue_file.endswith('indices'))
    with open(dialogue_file) as l:
        for line in l:
            dialogue_indices.append(ast.literal_eval(line.strip()))
    return dialogue_indices

def calc_dial_turn_idxs(dialogue_file):
    dialogue_indices = []
    full_dials = {}
    dialogue_index = -1
    turn_index = -1
    with open(dialogue_file) as l:
        for line in l:
            if line.startswith('#S'):
                dialogue_index += 1
                turn_index = 0
            else:
                dialogue_indices.append((dialogue_index, turn_index))
                turn_index += 1
    return dialogue_indices

def read_in_chat(chat_file, dialogues):
    chats = {}
    with open(chat_file) as c:
        for line in c:
            if line.startswith('dia'):
                continue
            else:
                line = line.strip().split(',')
                this_index = (int(line[0]), int(line[1]))
                # print(dialogues)
                chats[this_index] = (line[-2], line[-1])
    return chats

def print_test_features(tensor, confidence, ave_probs, ave_logprobs, target, dialogue_indices, labels, inv_labels, indices, fold_id, full_dials, feature_file):
    # dial_id, turn_id, predicted_label, correct_bool, prob, entropy, confidence, chat_prob, chat_rank
    tensor = torch.exp(tensor)
    probs, predicted = torch.max(tensor, 1)
    predicted = predicted.view(target.size()).data
    probs = probs.view(target.size()).data
    corrects = predicted == target.data
    confidence = confidence.squeeze().data.cpu().numpy() / 2
    ave_logprobs = ave_logprobs.squeeze().data.cpu().numpy() / 2
    ave_probs = ave_probs.squeeze().data.cpu().numpy() / 2
    tensor = tensor.squeeze().data.cpu().numpy()
    start_id, end_id = indices[fold_id]
    for ind, val in enumerate(corrects):
        item = []
        item_id = start_id+ind
        dialogue_index, turn_index = dialogue_indices[item_id]
        turn = full_dials[(dialogue_index, turn_index)]
        cs_idx = inv_labels[turn[3]] if turn[3] != "none" else None
        item.append(dialogue_index)
        item.append(turn_index)
        item.append(labels[predicted[ind]])
        item.append(str(bool(val)))
        item.append(probs[ind])
        if probs[ind] < 0.0:
            print(tensor[ind])
            print(probs[ind], predicted[ind])
            raise Exception
        item.append(stats.entropy(tensor[ind]))
        item.append(confidence[ind, predicted[ind]])
        item.append(ave_probs[ind, predicted[ind]])
        item.append(ave_logprobs[ind, predicted[ind]])
        item.append(tensor[ind, cs_idx] if cs_idx is not None else '') #cs_prob
        item.append(stats.rankdata(-tensor[ind], method='min')[cs_idx] if cs_idx is not None else '') #cs_rank
        print(','.join([str(x) for x in item]), file=feature_file)





