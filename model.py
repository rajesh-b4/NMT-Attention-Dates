# This file contains encoder-decoder seq2seq model with attention mechanism. The model is implemented in PyTorch. The model is trained on a dataset containing date strings in the human-readable format and expected output is string "YYYY-MM-DD". The model is trained to convert human-readable date strings to machine-readable format. 

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, emb_dim, enc_hid_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(input_vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True, batch_first = True)
        
        
    def forward(self,x):
        
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) # concatenate the hidden states of the forward and backward RNNs
        return outputs, hidden
    
class BahdanauAttention(nn.Module):
    """
    Use Bahdanau Attention formula is  
    e_ij = v^T * tanh(W_a * s_{i-1} + U_a * h_j) 
    where s_{i-1} is the previous hidden state of the decoder and h_j is the hidden state of the encoder
    
    alpha_ij = softmax(e_ij)

    """
    
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.W_a = nn.Linear(dec_hid_dim, dec_hid_dim)
        self.U_a = nn.Linear(enc_hid_dim*2, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
            
            # hidden = [batch_size, dec_hid_dim]
            # encoder_outputs = [batch_size, seq_len, enc_hid_dim*2]
                        
            projected_hidden = self.W_a(hidden.unsqueeze(1)) # [batch_size, 1, dec_hid_dim]
        
            energy = (torch.tanh(projected_hidden + self.U_a(encoder_outputs))) # [batch_size, seq_len, dec_hid_dim]
            
            
            attention = self.v(energy).squeeze(2) # [batch_size, seq_len]
            
            attention_weights = torch.softmax(attention, dim = 1) # [batch_size, seq_len]
            
            return attention_weights

class ConcatAttention(nn.Module):
    """
    Use Concatenative Attention formula is
    e_ij = v^T * tanh(W_a *[s_{i-1}; h_j])
    where s_{i-1} is the previous hidden state of the decoder and h_j is the hidden state of the encoder
    """
    
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.W_a = nn.Linear((enc_hid_dim*2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
            
            # hidden = [batch_size, dec_hid_dim]
            # encoder_outputs = [batch_size, seq_len, enc_hid_dim*2]
            
            hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1) # [batch_size, seq_len, dec_hid_dim]
            
            energy = torch.tanh(self.W_a(torch.cat((hidden, encoder_outputs), dim = 2))) # [batch_size, seq_len, dec_hid_dim]
            
            attention = self.v(energy).squeeze(2) # [batch_size, seq_len]
            
            attention_weights = torch.softmax(attention, dim = 1) # [batch_size, seq_len]
            
            return attention_weights

class Decoder(nn.Module):
    
    """
    Decoder with attention mechanism
    """
    def __init__(self, output_vocab_size, emb_dim, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.output_vocab_size = output_vocab_size
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        # self.concat_attention = ConcatAttention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(output_vocab_size, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim*2) + emb_dim, dec_hid_dim, batch_first = True) # we are passing the context vector and the embedded token as input by concatenating them
        self.fc = nn.Linear(dec_hid_dim, output_vocab_size) 
        
        
    def forward(self, x, hidden, encoder_outputs):
            
            # x = [batch_size]
            # hidden = [batch_size, dec_hid_dim]
            # encoder_outputs = [batch_size, seq_len, enc_hid_dim*2]
            
            x = x.unsqueeze(1) # [batch_size, 1]
            embedded = self.embedding(x) # [batch_size, 1, emb_dim]
            
            attention_weights = self.attention(hidden, encoder_outputs) # [batch_size, seq_len]
            # attention_weights = self.concat_attention(hidden, encoder_outputs) # [batch_size, seq_len]
            attention_weights = attention_weights.unsqueeze(1) # [batch_size, 1, seq_len]
            
            context_vector = torch.bmm(attention_weights, encoder_outputs) # [batch_size, 1, enc_hid_dim*2]
            
            rnn_input = torch.cat((embedded, context_vector), dim = 2) # [batch_size, 1, (enc_hid_dim*2) + emb_dim]
            
            output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0)) # output = [batch_size, 1, dec_hid_dim], hidden = [1, batch_size, dec_hid_dim]
            
            prediction = self.fc(output.squeeze(1)) # [batch_size, output_vocab_size]
            
            return prediction, hidden.squeeze(0), attention_weights.squeeze(1)
            
        
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, input, target, teacher_forcing_ratio = 0.5):
        
        # input = [batch_size, seq_len]
        # target = [batch_size, seq_len]
        
        batch_size = input.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_vocab_size
        
        outputs = torch.zeros(batch_size, target_len-1, target_vocab_size).to(self.device)
        
        attention_scores = torch.zeros(batch_size, target_len-1, input.shape[1]).to(self.device)
        
        
        encoder_outputs, hidden = self.encoder(input) 
        
        x = target[:,0] # <SOS> token
        
        for t in range(1, target_len):
            
            output, hidden, attention_weights = self.decoder(x, hidden, encoder_outputs)
            
            attention_scores[:,t-1] = attention_weights
            
            outputs[:,t-1] = output 
            
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            # x = target[:,t] if teacher_force else top1 # if teacher_force is True, we use the actual target token, else we use the predicted token
            x = top1 # we are not using teacher forcing
        return outputs, attention_scores
    
        