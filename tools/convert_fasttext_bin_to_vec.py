
try:
    import fasttext
except ImportError:
    print("Please install fasttext: pip install fasttext-wheel")
    exit(1)

import sys

def convert_bin_to_vec(input_path, output_path):
    print(f"Loading {input_path} with fasttext...")
    # fasttext.load_model uses mmap on Linux/Mac, might load on Windows but is generally more efficient than gensim for .bin
    model = fasttext.load_model(input_path)
    
    words = model.get_words()
    print(f"Loaded model. Vocabulary size: {len(words)}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header: number of words and dimension
        dim = model.get_dimension()
        f.write(f"{len(words)} {dim}\n")
        
        for i, word in enumerate(words):
            if i % 10000 == 0:
                print(f"Processed {i} words...", end="\r")
            vec = model.get_word_vector(word)
            vec_str = " ".join([f"{v:.6f}" for v in vec])
            f.write(f"{word} {vec_str}\n")
            
    print(f"\nSaved vectors to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_fasttext_bin_to_vec.py <input.bin> <output.vec>")
    else:
        convert_bin_to_vec(sys.argv[1], sys.argv[2])
