import json, re
from pathlib import Path
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent.absolute()
ROOT = SCRIPT_DIR.parent
JSON = ROOT / "jsons"
BAD = re.compile(r"as an ai( language)? model|\[(your|insert|name|company)[^\]]*\]", re.I)

def ok(msgs):
    core = msgs[1:] if msgs[0]["role"] == "system" else msgs
    if len(core) < 2 or core[-1]["role"] != "assistant":
        return False
    if any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(core)):
        return False                                   # roles must strictly alternate
    if any(not m["content"] for m in msgs):
        return False
    return not any(BAD.search(m["content"]) for m in msgs if m["role"] == "assistant")

def pull(name, cfg, n):
    ds = load_dataset(name, cfg, split="train").shuffle(seed=0)
    out = []
    for r in ds:
        msgs = [{"role": m["role"], "content": m["content"].strip()} for m in r["messages"]]
        if msgs and ok(msgs):
            out.append({"messages": msgs})
            if len(out) >= n:
                break
    print(name, cfg, len(out))
    return out

if __name__ == "__main__":
    chat = (pull("HuggingFaceTB/smoltalk", "everyday-conversations", 2500)
          + pull("HuggingFaceTB/smoltalk", "systemchats-30k", 5000)
          + pull("HuggingFaceTB/smoltalk", "smol-magpie-ultra", 15000))
    json.dump(chat, open(JSON / "chat.json", "w", encoding="utf-8"))