# Lets include the function to build the vocabulary here

from collections import defaultdict

def build_vocab(file_path):
    
    """
    args:
    file_path: path to the dataset file
    
    returns:
    input_vocab: dictionary containing character to index mapping for input language
    output_vocab: dictionary containing character to index mapping for output language
    input_vocab_inv: dictionary containing index to character mapping for input language
    output_vocab_inv: dictionary containing index to character mapping for output language
    
    """
    
    input_vocab = defaultdict(lambda:len(input_vocab))
    output_vocab = defaultdict(lambda:len(output_vocab))
    
    input_vocab['<PAD>'] = 0
    input_vocab['<UNK>'] = 1
    output_vocab['<PAD>'] = 0
    output_vocab['<UNK>'] = 1
    
    output_vocab['<SOS>'] = 2
    output_vocab['<EOS>'] = 3
    
    
    with open(file_path, 'r') as file:
        for line in file:
            input_sent, output_sent = line.strip().split(',')
            input_sent = input_sent.strip().strip("'")
            output_sent = output_sent.strip().strip("'")
            
            for char in input_sent:
                input_vocab[char]
                
            for char in output_sent:
                output_vocab[char]
                
                
    input_vocab = dict(input_vocab)
    output_vocab = dict(output_vocab)
    
    input_vocab_inv = {v:k for k,v in input_vocab.items()}
    output_vocab_inv = {v:k for k,v in output_vocab.items()}
    
    return input_vocab, output_vocab, input_vocab_inv, output_vocab_inv


# Use BPE tokenization to build the vocabulary

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    import re
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

