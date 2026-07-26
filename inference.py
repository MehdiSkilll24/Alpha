import torch
from main import Transformer
from tokenizer import Tokenizer
import json
from safetensors.torch import load_file

with open("config.json") as f:
    config = json.load(f)


def rebuild_tokenizer(word_to_idx):
    tok = Tokenizer()

    tok.word_to_idx = word_to_idx

    tok.idx_to_word = {
        int(idx): word
        for word, idx in word_to_idx.items()
    }

    return tok


def translate(model, src_tokenizer, tgt_tokenizer, sentence, device, max_len=50):
    model.eval()

    src_ids = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor([src_ids]).to(device)

    decoder_input = torch.tensor(
        [[tgt_tokenizer.word_to_idx["<bos>"]]]
    ).to(device)

    with torch.no_grad():

        for _ in range(max_len):

            logits = model(src_tensor, decoder_input)

            next_token = logits[:, -1, :].argmax(dim=-1).item()

            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.tensor([[next_token]]).to(device)
                ],
                dim=1
            )

            if next_token == tgt_tokenizer.word_to_idx["<eos>"]:
                break

    output_ids = decoder_input[0].tolist()

    return tgt_tokenizer.decode(output_ids)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    # Load vocabularies
    with open("src_vocab.json", "r", encoding="utf-8") as f:
        src_vocab = json.load(f)

    with open("tgt_vocab.json", "r", encoding="utf-8") as f:
        tgt_vocab = json.load(f)


    # Rebuild tokenizers
    src_tokenizer = rebuild_tokenizer(src_vocab)
    tgt_tokenizer = rebuild_tokenizer(tgt_vocab)


    # Load config
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)


    # Build model
    model = Transformer(
    src_vocab_size=config["src_vocab_size"],
    tgt_vocab_size=config["tgt_vocab_size"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    num_decoder_layers=config["num_encoder_layers"],
    num_encoder_layers=config["num_decoder_layers"],
    d_ff=config["d_ff"]
    ).to(device)


    # Load weights
    weights = load_file(
        "model.safetensors"
    )

    model.load_state_dict(weights)

    model.eval()


    print("Type an English sentence (or 'quit' to exit):")

    while True:

        sentence = input("> ")

        if sentence.lower() == "quit":
            break

        translation = translate(
            model,
            src_tokenizer,
            tgt_tokenizer,
            sentence,
            device
        )

        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()