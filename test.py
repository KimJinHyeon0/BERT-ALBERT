import torch
from transformers import BertTokenizer, BertModel
from transformers import AlbertTokenizer, AlbertModel
import time

cuda = torch.device('cuda')
time_0 = time.time()
text = "Hello, my dog is cute."
print("Sample Text :\n", text)

##Bert
print("\nExcute Bert")
tokenizer_Bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_Bert = BertModel.from_pretrained("bert-base-uncased").cuda()
encoded_input_Bert = tokenizer_Bert(text, return_tensors='pt').cuda()
output_Bert = model_Bert(**encoded_input_Bert)
time_1 = time.time()
print(output_Bert.last_hidden_state)
print(output_Bert.last_hidden_state.shape)
print("time_Bert : %.5fs" %(time_1 - time_0))


#Albert
print("\nExcute Albert")
tokenizer_Albert = AlbertTokenizer.from_pretrained('albert-base-v2')
model_Albert = AlbertModel.from_pretrained('albert-base-v2').cuda()
encoded_input_Albert = tokenizer_Albert(text, return_tensors='pt').cuda()
output_Albert = model_Albert(**encoded_input_Albert)
time_2 = time.time()
print(output_Albert.last_hidden_state)
print(output_Albert.last_hidden_state.shape)
print("time_Albert : %.5fs" %(time_2 - time_1))