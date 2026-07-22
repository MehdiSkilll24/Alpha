import torch
from main import Transformer
from tokenizer import Tokenizer

CKPT_PATH = r"C:\Users\mehdi\Desktop\Pythonfiles\Projects\Transformers\Alpha\checkpoint_epoch3_q4.pt"

def rebuild_tokenizer(word_to_idx):
    tok = Tokenizer()
    tok.word_to_idx = word_to_idx
    tok.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return tok

def translate(model, src_tokenizer, tgt_tokenizer, sentence, device, max_len=50):
    model.eval()

    src_ids = src_tokenizer.encode(sentence)
    src_tensor = torch.tensor([src_ids]).to(device)

    decoder_input = torch.tensor([[tgt_tokenizer.word_to_idx["<bos>"]]]).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src_tensor, decoder_input)
            next_token = logits[:, -1, :].argmax(dim=-1).item()

            decoder_input = torch.cat(
                [decoder_input, torch.tensor([[next_token]]).to(device)],
                dim=1
            )

            if next_token == tgt_tokenizer.word_to_idx["<eos>"]:
                break

    output_ids = decoder_input[0].tolist()
    return tgt_tokenizer.decode(output_ids)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CKPT_PATH, map_location=device)

    src_tokenizer = rebuild_tokenizer(checkpoint["src_vocab"])
    tgt_tokenizer = rebuild_tokenizer(checkpoint["tgt_vocab"])

    model = Transformer(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Type an English sentence (or 'quit' to exit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == "quit":
            break

        translation = translate(model, src_tokenizer, tgt_tokenizer, sentence, device)
        print(f"Translation: {translation}")


if __name__ == "__main__":
    main()