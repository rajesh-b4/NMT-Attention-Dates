from vocab import build_vocab
from model import Encoder, Decoder, Seq2Seq
import torch

input_vocab, output_vocab, input_vocab_inv,output_vocab_inv = build_vocab('Data/train.txt')
input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
max_input_len = 20
max_output_len = 10
batch_size = 32
embedding_size = 128
enc_hidden_size = 128
dec_hidden_size = 2*128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(input_vocab_size, embedding_size, enc_hidden_size)
decoder = Decoder(output_vocab_size, embedding_size, enc_hidden_size, dec_hidden_size)
model = Seq2Seq(encoder, decoder, device).to(device)

model.load_state_dict(torch.load('Models/model4.pth'))

# Now load validation.txt  where each line is a pair of input and output date separated by a comma, so give input to the predict function and compare whether the output is correct or not
def predict(model,src,src_vocab,tgt_vocab,tgt_inv_vocab,max_len,device):
    
 
    src = torch.tensor([src_vocab.get(char,src_vocab['<UNK>']) for char in src]).unsqueeze(0).to(device)
    
    tgt = [tgt_vocab['<SOS>']]+[tgt_vocab['<PAD>']]*max_len+[tgt_vocab['<EOS>']]
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    
    outputs,attention_scores = model(src,tgt,0)
    
    outputs = outputs.squeeze(0)
   
    decoder_outputs = []
    for output in outputs:
            output = output.argmax(0).item()
            
            if output == tgt_vocab['<EOS>']:
                break
            decoder_outputs.append(tgt_inv_vocab[output])
            # decoder_outputs.append(output)
    return "".join(decoder_outputs)
    
    
actual_outputs = []
predicted_outputs = []

with open('Data/validation.txt', 'r') as file:
    for line in file:
        input_sent, output_sent = line.strip().split(',')
        input_sent = input_sent.strip().strip("'")
        output_sent = output_sent.strip().strip("'")
        
        output = predict(model, input_sent, input_vocab, output_vocab, output_vocab_inv, max_output_len, device)
        
        actual_outputs.append(output_sent)
        predicted_outputs.append(output)
        
        
        
print('All tests passed')

# save these actual and  predicted in  a txt file
with open('Data/predictions.txt', 'w') as file:
    for actual, predicted in zip(actual_outputs, predicted_outputs):
        file.write(actual + ',' + predicted + '\n')
        

# save all the actual and predicted outputs in a list and return it as i nedd to perform some analysis on it

"""

calculate Average Validation Set Error in % (using "Exact Match over all 10 outputs" as a metric),
Average Validation Set Error in % (number of mismatches averaged over all 10 outputs)(since the ouput is always 10 characters long),
Numbering the outputs from 1 to 10 (1 for the most significant digit of the year and 10 for the least significant digit of the date), the validation set error (average number of mismatches) for which output was the highest?
Numbering the outputs from 1 to 10 (1 for the most significant digit of the year and 10 for the least significant digit of the date), the validation set error (average number of mismatches) for which output was the lowest?
"""

def calculate_all_errors(actual_outputs, predicted_outputs):
    
    exact_match_error = 0
    mismatch_error = 0
    position_errors = [0]*10
    
    less_than_10 = 0
    
    for actual, predicted in zip(actual_outputs, predicted_outputs):
        
        if len(actual) != 10 or len(predicted) != 10:
            less_than_10 += 1
            continue
        
        exact_match_error += 1 if actual == predicted else 0
        for i in range(10):
            mismatch_error += 1 if actual[i] != predicted[i] else 0
            position_errors[i] += 1 if actual[i] != predicted[i] else 0
            
    highest_error = position_errors.index(max(position_errors)) + 1
    lowest_error = position_errors.index(min(position_errors)) + 1
    
    print("Excat matches ", exact_match_error)
    print("Less than 10 ", less_than_10)
        
    exact_match_error = (exact_match_error/len(actual_outputs))*100
    mismatch_error = (mismatch_error/(len(actual_outputs)*10))*100
    
    return exact_match_error, mismatch_error, highest_error, lowest_error

exact_match_error, mismatch_error, highest_error, lowest_error = calculate_all_errors(actual_outputs, predicted_outputs)


print('Exact Match Error:', exact_match_error)
print('Mismatch Error:', mismatch_error)
print('Highest Error:', highest_error)
print('Lowest Error:', lowest_error)
