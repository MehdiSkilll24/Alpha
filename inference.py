import torch
from main import Transformer
from tokenizer import Tokenizer
import json
import os
from tokenizers import Tokenizer as HFTokenizer
from bpe_tokenizer import BPETokenizerWrapper


def rebuild_tokenizer(word_to_idx):
    tok = Tokenizer()
    tok.word_to_idx = word_to_idx
    tok.idx_to_word = {
        int(idx): word
        for word, idx in word_to_idx.items()
    }
    return tok

def translate(model, tokenizer, sentence, device, max_len=50):
    model.eval()
    src_ids = tokenizer.encode(sentence)
    src_tensor = torch.tensor([src_ids]).to(device)

    decoder_input = torch.tensor(
        [[tokenizer.word_to_idx["<bos>"]]]
    ).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src_tensor, decoder_input)
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_token]]).to(device)],
                dim=1
            )

            if next_token == tokenizer.word_to_idx["<eos>"]:
                break

    output_ids = decoder_input[0].tolist()
    return tokenizer.decode(output_ids)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(script_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    
    hf_tok = HFTokenizer.from_file(os.path.join(script_dir, "bpe_tokenizer.json"))
    tokenizer = BPETokenizerWrapper(hf_tok)

    actual_vocab_size = len(tokenizer)
    print(f"BPE vocab size: {actual_vocab_size}")

    model = Transformer(
        actual_vocab_size,
        actual_vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_encoder_layers=config.get("num_encoder_layers", 0),
        num_decoder_layers=config["num_decoder_layers"],
        model_type=config.get("model_type", "decoder-only")
    ).to(device)

    x = sum(p.numel() for p in model.parameters())
    print(f"Total params {x}")

    #checkpoint = torch.load(os.path.join(script_dir, "checkpoints", "checkpoint_latest.pt"), map_location=device)
    #model.load_state_dict(checkpoint["model_state_dict"])
    #model.eval()
#
    #print("Type an English sentence (or 'quit' to exit):")
    #while True:
    #    sentence = input("> ")
    #    if sentence.lower() == "quit":
    #        break
    #    translation = translate(model, tokenizer, sentence, device)
    #    print(f"Translation: {translation}")


if __name__ == "__main__":
    main()