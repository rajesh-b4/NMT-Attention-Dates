import torch
from torch.utils.data import Dataset, DataLoader

from vocab import build_vocab

class DateTranslationDataset(Dataset):
    def __init__(self, file_path,input_vocab, output_vocab,max_input_len, max_output_len=10):
        
        """
        Args:
        file_path: path to the data file
        input_vocab: input vocabulary
        output_vocab: output vocabulary
        max_input_len: maximum length of input sequence
        max_output_len: maximum length of output sequence is always 10 because ouput format is YYYY-MM-DD
        """
        
        self.input_vocab = input_vocab  
        self.output_vocab = output_vocab
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.data = self.load_data(file_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])
    
    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                input_sent, output_sent = line.strip().split(',')
                input_sent = input_sent.strip().strip("'")
                output_sent = output_sent.strip().strip("'")
                
                input_ids = [self.input_vocab.get(char, self.input_vocab['<UNK>']) for char in input_sent][:self.max_input_len]
                output_ids = [self.output_vocab.get(char, self.output_vocab['<UNK>']) for char in output_sent][:self.max_output_len]
                
                # padding on left side
                input_ids = [self.input_vocab['<PAD>']]*(self.max_input_len - len(input_ids)) + input_ids
                output_ids = [self.output_vocab['<PAD>']]*(self.max_output_len - len(output_ids)) + output_ids
                
                output_ids = [self.output_vocab['<SOS>']] + output_ids + [self.output_vocab['<EOS>']]
                
                data.append((input_ids, output_ids))
                
        return data
                
               
                
def get_dataloader(file_path, input_vocab, output_vocab, max_input_len, max_output_len, batch_size):
    dataset = DateTranslationDataset(file_path, input_vocab, output_vocab, max_input_len, max_output_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# # Lets test the dataloader
# input_vocab, output_vocab, _, _ = build_vocab('Data/train.txt')
# dataloader = get_dataloader('Data/train.txt', input_vocab, output_vocab, 20, 10, 2)

# for input_batch, output_batch in dataloader:
    
#     print('Input Batch Shape:', input_batch.shape)
#     print('Output Batch Shape:', output_batch.shape)
    
#     # Lets print the first batch
#     print('Input Batch:', input_batch)
    
#     print('Output Batch:', output_batch)
    
#     break

    
        
       