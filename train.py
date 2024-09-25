from vocab import build_vocab
from dataloader import get_dataloader
from model import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

def train(model,trainloader,epochs,optimizer,criterion,device):
    
    model.train()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        print('*'*20 + f'Epoch {epoch+1}' + '*'*20)
        for src,tgt in trainloader:
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            output,_ = model(src,tgt)
            
            tgt = tgt[:,1:]
            
            output_dim = output.shape[-1]
            output = output.reshape(-1,output_dim)
            tgt = tgt.reshape(-1)
    
            loss = criterion(output,tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch: {epoch+1:02}')
        
        
        print(f'Train Loss: {epoch_loss/len(trainloader):.3f}')
        print(f'Validation Loss: {evaluate(model,validloader,criterion,device):.3f}')
        
    # Save the model in Models folder
    torch.save(model.state_dict(), 'Models/model4.pth')
    

def evaluate(model,validloader,criterion,device):
    model.eval()
    epoch_loss = 0
    accuracy = 0 
    total = 0
    with torch.no_grad():
        for src,tgt in validloader:
            src = src.to(device)
            tgt = tgt.to(device)
            output,_ = model(src,tgt,0) #turn off teacher forcing  
            output_dim = output.shape[-1]
            output = output.reshape(-1,output_dim)
            tgt = tgt[:,1:]
            tgt = tgt.reshape(-1)
            loss = criterion(output,tgt)
            
           # check if all the characters are correct , if yes then increment the accuracy
            # print(torch.argmax(output,dim=1).shape,tgt.shape)
            accuracy += torch.sum(torch.argmax(output,dim=1) == tgt).item()
            total += tgt.shape[0]
            epoch_loss += loss.item()
            
    print(f'Validation Accuracy: {accuracy/total:.3f}')
        
    return epoch_loss/len(validloader)

def predict(model,src,src_vocab,tgt_vocab,tgt_inv_vocab,max_len,device):
    
 
    src = torch.tensor([src_vocab.get(char,src_vocab['<UNK>']) for char in src]).unsqueeze(0).to(device)
    
    tgt = [tgt_vocab['<SOS>']]+[tgt_vocab['<PAD>']]*max_len+[tgt_vocab['<EOS>']]
    tgt = torch.tensor(tgt).unsqueeze(0).to(device)
    
    outputs,attention_scores = model(src,tgt,0)
    
    outputs = outputs.squeeze(0)
    
    print(outputs.shape)
    decoder_outputs = []
    for output in outputs:
            output = output.argmax(0).item()
            
            if output == tgt_vocab['<EOS>']:
                break
            decoder_outputs.append(tgt_inv_vocab[output])
            # decoder_outputs.append(output)
    # return "".join(decoder_outputs)
    return decoder_outputs,attention_scores



input_vocab, output_vocab, input_vocab_inv,output_vocab_inv = build_vocab('Data/train.txt')
input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
max_input_len = 16
max_output_len = 10
batch_size = 32
embedding_size = 128 # 128 used for model with accuracy 0.88
enc_hidden_size = 128  #use 128 for model with accuracy 0.88
dec_hidden_size = 2*128 #use 2*128 for model with accuracy 0.88
learning_rate = 0.0015
num_epochs = 3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader = get_dataloader('Data/train.txt', input_vocab, output_vocab, max_input_len, max_output_len, batch_size)
validloader = get_dataloader('Data/validation.txt', input_vocab, output_vocab, max_input_len, max_output_len, batch_size)


encoder = Encoder(input_vocab_size, embedding_size, enc_hidden_size)
decoder = Decoder(output_vocab_size, embedding_size, enc_hidden_size, dec_hidden_size)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index = output_vocab['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model,trainloader,num_epochs,optimizer,criterion,device)

model.load_state_dict(torch.load('Models/model4.pth'))

def attention_visualization(model,src,input_vocab,output_vocab,output_vocab_inv,max_output_len,device):
    outputs,attention_scores = predict(model,src,input_vocab,output_vocab,output_vocab_inv,max_output_len,device)
    src_tokens = [char for char in src]
    tgt_tokens = outputs
    
    #convert attention scores to numpy
    
    attention_scores = attention_scores.squeeze(0).cpu().detach().numpy() # [tgt_len, src_len]
    
   
    
    print('Source:', src)
    print('Predicted:', "".join(outputs))
    
    
    fig, ax = plt.subplots(figsize=(12,12))
    cax=ax.matshow(attention_scores, cmap='bone')
    
    ax.set_xticks(np.arange(len(src_tokens)))
    ax.set_yticks(np.arange(len(tgt_tokens)))
  
    
    ax.set_xticklabels(src_tokens, rotation=90,)
    ax.set_yticklabels(tgt_tokens)
    
    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')
    
    fig.colorbar(cax)
    
    plt.show()
    
    #save the plot
    
    plt.savefig('plots/attention4.png')
    
    

attention_visualization(model,'29 March 2022',input_vocab,output_vocab,output_vocab_inv,max_output_len,device)





