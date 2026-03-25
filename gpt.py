import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 16
n_head = 4
n_layer = 6
dropout = 0.2
# -------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X.to(device), Y.to(device))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class PositionalEncoding(nn.Module):
    """
    Create a Positional Encoding Layer
    """

    def __init__(self):
        super().__init__()

    # Recieve a 2d matrix
    def forward(self, input_tensor: torch.tensor) -> torch.tensor:
        self.num_of_tokens = input_tensor.shape[1]
        self.d_model = input_tensor.shape[2]
        vector1 = torch.arange(self.num_of_tokens).reshape(-1, 1).float()
        # Vector from 0 to n-1 Shape(n, 1)

        vector2 = torch.arange(self.d_model/2).reshape(-1, 1)
        vector3 = torch.e**(-2*vector2*math.log(10000)/self.d_model).T
        # Vector for division terms Shape(1, d_model)

        matrix = torch.matmul(vector1, vector3)
        # Matrix with shape(num_of_tokens, d_model/2)

        cosine_matrix = torch.cos(matrix)
        sine_matrix = torch.sin(matrix)
        
        # 1. Create an empty matrix of shape (num_of_tokens, d_model)
        pe_matrix = torch.zeros(self.num_of_tokens, self.d_model)

        # 2. Fill the even columns (0, 2, 4...) with sine_matrix
        pe_matrix[:, 0::2] = sine_matrix

        # 3. Fill the odd columns (1, 3, 5...) with cosine_matrix
        pe_matrix[:, 1::2] = cosine_matrix
        
        output_tensor = input_tensor + pe_matrix

        return output_tensor

class MaskedMultiHeadAttention(nn.Module):
    """
    Compute Masked Dot Product Attention using Multiple Heads
    """

    def __init__(self, d_model:int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=False)

        # Checking if d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads # Dimension of query, key and value vectors within each head

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Checking if input has correct embedding dimension
        if input.shape[-1] != self.d_model:
            raise ValueError(f"Expected input feature dimension to be d_model ({self.d_model}), but got {input.shape[-1]}")

        # Input Dimension: (batch_size, num_tokens, d_model)
        batch_size = input.shape[0]
        num_tokens = input.shape[1]

        # Creating query, key and value vectors from input embeddings. 
        # These vectors contain queries from all the heads combined.
        q = self.w_q(input)
        k = self.w_k(input)
        v = self.w_v(input)
        # Dimension: (batch_sizee, num_tokens, d_model)

        # Splitting our vectors into heads by reshaping our tensors.
        q_head = q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k_head = k.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v_head = v.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        # Dimension: (batch_size, num_tokens, num_heads, head_dimension)

        # Swapping (num_tokens) and (num_heads) dimensions.
        # We do this because we need (num_tokens) and (head_dimension) at the end.
        # This is because we need to perform matrix operations on the last two dimensions
        q_head = q_head.transpose(1, 2)
        k_head = k_head.transpose(1, 2)
        v_head = v_head.transpose(1, 2)
        # Dimension: (batch_size, num_heads, num_tokens, head_dimension)

        # Getting a transpose of key vectors for matrix multiplication 
        k_t = k_head.transpose(-2, -1)
        sim1 = (q_head @ k_t)/math.sqrt(self.head_dim) # Scaled dot product of query and key vectors
        # sim1 is the matrix of attention scores for each token
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Creating a matrix where lower triangle is 1, rest is 0
        mask = torch.tril(torch.ones(num_tokens, num_tokens))
        # Dimension: (num_queries, num_keys)

        # Applying the mask to our attention scores
        sim1 = sim1.masked_fill(mask == 0, float('-inf'))

        # Performing softmax along the keys dimension to get attention scores for each key
        sim2 = F.softmax(sim1, dim=-1)
        # Dimension: (batch_size, num_heads, num_queries, num_keys)

        # Weighted sum of attention weights and value vectors
        sim3 = sim2 @ v_head
        # Dimension: (batch_size, num_heads, num_queries, head_dimension)

        # Swapping (num_heads) and (num_queries) dimension back so that we can recombine all the heads
        sim3 = sim3.transpose(1, 2).contiguous()
        sim3 = sim3.view(batch_size, num_tokens, self.d_model)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        # Passing all of our heads into a final layer so that all heads can interact.
        output = self.w_o(sim3)
        # Dimension: (batch_size, num_tokens, d_model)
        # Same as input dimension
        
        return output

class FFNN(nn.Module):
    """
    Apply position-wise feed-forward network.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected input feature dimension to be d_model ({self.d_model}), but got {x.shape[-1]}")

        z1 = self.hidden_layer(x)
        a1 = self.activation(z1)
        a1 = self.dropout(a1) # Randomly zeroes out some neurons to prevent overfitting
        output = self.output_layer(a1)

        return output
    
class GPT(nn.Module):
    """ One layer of GPT with attention and FFNN """

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.ffnn = FFNN(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dimension of x: (Batch_size, num_tokens, d_model)
        x = x + self.dropout(self.attention(self.norm1(x))) # Residual connection with Pre Normalization
        x = x + self.dropout(self.ffnn(self.norm2(x))) # Residual connection with Pre Normalization
        # Dimension: (Batch_size, num_tokens, d_model)
        return x
    
class GPTStack(nn.Module):
    """ A stack of GPT Layers """

    def __init__(self, vocab_size: int, d_model: int, num_blocks: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model     
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding()
        self.layers = nn.ModuleList([GPT(d_model, num_heads, hidden_dim, dropout_rate) for _ in range(num_blocks)])
        self.final_output_logits = nn.Linear(d_model, vocab_size)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        # Dimension of tokens: (Batch_size, num_tokens)

        x = self.embedding(x)
        x = self.positional_encoding(x)
        # Dimension: (Batch_size, num_tokens, d_model)

        for layer in self.layers:
            x = layer(x)
        
        norm_output = self.norm1(x)

        logits = self.final_output_logits(norm_output)
        # Dimension: (Batch_size, num_tokens, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Reshaping our Tensors
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:

        for _ in range(max_new_tokens):

            x_cond = x[:, -block_size:]

            logits, loss = self(x_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            x_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, x_next), dim=1)

        return x
    
model = GPTStack(vocab_size, n_embd, n_layer, n_head, n_embd * 4, dropout)
m = model.to(device)

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluage the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
