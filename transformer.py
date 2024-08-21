import math
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).reshape(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * -math.log(10000.0) / dim_model
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)  # If is even
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)  # If is odd
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_encoding[: token_embedding.size(0), :]
        )


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.fc = nn.Linear(dim_model, num_tokens)

    def forward(
        self,
        source,
        target,
        target_mask=None,
        source_pad_mask=None,
        target_pad_mask=None,
    ):
        source = self.embedding(source) * math.sqrt(self.dim_model)
        target = self.embedding(target) * math.sqrt(self.dim_model)
        source = self.positional_encoder(source).permute(1, 0, 2)
        target = self.positional_encoder(target).permute(1, 0, 2)
        transformer_out = self.transformer(
            source,
            target,
            tgt_mask=target_mask,
            src_key_padding_mask=source_pad_mask,
            tgt_key_padding_mask=target_pad_mask,
        )
        out = self.fc(transformer_out)
        return out


def get_target_mask(size):
    mask = torch.tril(torch.ones(size, size) == 1).float()
    mask = mask.masked_fill(mask == 0, float("-inf"))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def create_pad_mask(matrix, pad_token):
    return matrix == pad_token


def generate_random_data(n):
    sos_token = np.array([2])
    eos_token = np.array([3])
    length = 8
    data = []
    for _ in range(n // 3):
        X = np.concatenate((sos_token, np.ones(length), eos_token))
        y = np.concatenate((sos_token, np.ones(length), eos_token))
        data.append([X, y])
    for _ in range(n // 3):
        X = np.concatenate((sos_token, np.zeros(length), eos_token))
        y = np.concatenate((sos_token, np.zeros(length), eos_token))
        data.append([X, y])
    for _ in range(n // 3):
        X = np.zeros(length)
        start = random.randint(0, 1)
        X[start::2] = 1
        y = np.zeros(length)
        if X[-1] == 0:
            y[0::2] = 1
        else:
            y[1::2] = 1
        X = np.concatenate((sos_token, X, eos_token))
        y = np.concatenate((sos_token, y, eos_token))
        data.append([X, y])
    np.random.shuffle(data)
    return data


def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        if idx + batch_size < len(data):
            if padding:
                max_batch_length = 0
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length
            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))
    print(f"{len(batches)} batches of size {batch_size}")
    return batches


train_dataloader = batchify_data(generate_random_data(9000))
valid_dataloader = batchify_data(generate_random_data(3000))

model = Transformer(
    num_tokens=4,
    dim_model=8,
    num_heads=2,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dropout_p=0.05,
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train_loop(model, optimizer, loss_fn, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        X = torch.tensor(batch[:, 0])
        y = torch.tensor(batch[:, 1])
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        target_mask = get_target_mask(y_input.size(1))
        pred = model(X, y_input, target_mask).permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            X = torch.tensor(batch[:, 0], dtype=torch.long)
            y = torch.tensor(batch[:, 1], dtype=torch.long)
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            target_mask = get_target_mask(y_input.size(1))
            pred = model(X, y_input, target_mask).permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
    return total_loss / len(dataloader)


def fit(model, optimizer, loss_fn, train_dataloader, valid_dataloader, epochs):
    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)
        train_loss = train_loop(model, optimizer, loss_fn, train_dataloader)
        validation_loss = validation_loop(model, loss_fn, valid_dataloader)
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()


fit(model, optimizer, loss_fn, train_dataloader, valid_dataloader, 10)


def predict(model, input_sequence, max_length=15, sos_token=2, eos_token=3):
    model.eval()
    y_input = torch.tensor([[sos_token]], dtype=torch.long)
    num_tokens = len(input_sequence[0])
    for _ in range(max_length):
        target_mask = get_target_mask(y_input.size(1))
        pred = model(input_sequence, y_input, target_mask)
        next_item = torch.tensor([[pred.topk(1)[1].reshape(-1)[-1].item()]])
        y_input = torch.cat((y_input, next_item), dim=1)
        if next_item.reshape(-1).item() == eos_token:
            break
    return y_input.reshape(-1).tolist()


examples = [
    torch.tensor([[2, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=torch.long),
    torch.tensor([[2, 1, 1, 1, 1, 1, 1, 1, 1, 3]], dtype=torch.long),
    torch.tensor([[2, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 3]], dtype=torch.long),
    torch.tensor([[2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 3]], dtype=torch.long),
    torch.tensor([[2, 0, 1, 3]], dtype=torch.long),
]

for i, example in enumerate(examples):
    result = predict(model, example)
    print(f"Example {i+1}")
    print(f"Input: {example.reshape(-1).tolist()[1:-1]}")
    print(f"Continuation: {result[1:-1]}")
    print()
