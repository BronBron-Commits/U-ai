import glob, os

def collect_corpora(base_dirs):
    texts = []
    for base in base_dirs:
        for path in glob.glob(os.path.join(base, '**/*.txt'), recursive=True):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read().strip()
                if data:
                    texts.append(data)
    return "\n".join(texts)

if __name__ == "__main__":
    dirs = ["../corpora", "../modules", "../data"]
    all_text = collect_corpora(dirs)
    with open("../data/train.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    print("✅ Combined dataset written to data/train.txt")
