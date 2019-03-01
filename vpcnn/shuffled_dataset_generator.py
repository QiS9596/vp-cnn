from chatscript_file_generator import *
import random

def shuffle_data(dataset_list, dialogues, shuffled_data_file, indices_file, dataset2_list = None, shuffled_data2 = None):
    len_dataset = len(dataset_list)
    indices = list(range(len_dataset))
    if dataset2_list is not None: assert len(dataset2_list) == len_dataset
    random.shuffle(indices)
    shuffled_data_file_handle = open(shuffled_data_file, 'w')
    indices_file_handle = open(indices_file, 'w')
    shuffled_data2_file_handle = None
    if dataset2_list is not None and shuffled_data2 is not None:
        shuffled_data2_file_handle = open(shuffled_data2, 'w')
    for index in indices:
        print(dataset_list[index].strip(), file=shuffled_data_file_handle)
        print(dialogues[index], file=indices_file_handle)
        if dataset2_list is not None and shuffled_data2 is not None:
            print(dataset2_list[index].strip(), file=shuffled_data2_file_handle)
    shuffled_data_file_handle.close()
    indices_file_handle.close()
    if shuffled_data2_file_handle is not None:
        shuffled_data2_file_handle.close()

def main(data_file, dialogue_file, shuffled_data_file, indices_file, dataset2 = None, shuffled2 = None):
    dialogues = calc_dial_turn_idxs(dialogue_file)
    data_list = open(data_file).readlines()
    data2_list = None
    if dataset2 is not None:
        data2_list = open(dataset2).readlines()
    shuffle_data(data_list, dialogues, shuffled_data_file, indices_file, data2_list, shuffled2)

if __name__ == '__main__':
    dialogue_file = 'corrected.tsv'
    data_file = 'vp16.base.stripped.lbl_in.tsv'
    data2_file = 'vp16.base.stripped.lbl_phn+bd.tsv'
    a = random.randint(0, 100)
    shuffled_data_file = 'vp16.base.word.shuffled.'+str(a)+'.txt'
    shuffled_data2_file = 'vp16.base.phn+bd.shuffled.'+str(a)+'.txt'
    indices_file = 'vp16.base.shuffled.'+str(a)+'.indices'
    main(data_file, dialogue_file,shuffled_data_file, indices_file, 
         dataset2=data2_file, 
         shuffled2=shuffled_data2_file)
