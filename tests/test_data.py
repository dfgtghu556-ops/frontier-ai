import numpy as np
import pytest
import torch

from frontier_ai.data.dataset import TokenDataset, write_tokens
from frontier_ai.data.synthetic import generate_corpus
from frontier_ai.data.tokenizer import CharTokenizer, Tokenizer, WordTokenizer


def test_char_tokenizer_roundtrip(tiny_text):
    tok = CharTokenizer.fit(tiny_text)
    ids = tok.encode(tiny_text)
    assert max(ids) < tok.vocab_size
    assert tok.decode(ids) == tiny_text


def test_word_tokenizer_roundtrip():
    text = "the cat sat. the cat slept, again!"
    tok = WordTokenizer.fit(text, min_count=1)
    assert tok.decode(tok.encode(text)) == text
    assert tok.stoi["<unk>"] == 0


def test_word_tokenizer_unknown_token_maps_to_unk():
    tok = WordTokenizer.fit("alpha beta", min_count=1)
    ids = tok.encode("gamma")
    assert ids == [tok.unk_id]


def test_tokenizer_save_load_roundtrip(tmp_path, tiny_text):
    tok = WordTokenizer.fit(tiny_text, min_count=1)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert isinstance(loaded, WordTokenizer)
    assert loaded.itos == tok.itos
    assert loaded.encode("the cat") == tok.encode("the cat")


def test_write_tokens_picks_compact_dtype(tmp_path):
    meta = write_tokens(tmp_path / "small.bin", list(range(100)) * 3, vocab_size=100, level="char")
    assert meta.dtype == "uint8"
    meta32 = write_tokens(tmp_path / "big.bin", [0, 1, 2], vocab_size=70_000, level="word")
    assert meta32.dtype == "int32"


def test_dataset_splits_and_batches(tmp_path, tiny_text):
    tok = CharTokenizer.fit(tiny_text)
    ids = tok.encode(tiny_text)
    bin_path = tmp_path / "tokens.bin"
    meta = write_tokens(bin_path, ids, tok.vocab_size, "char", val_frac=0.2)
    ds = TokenDataset(bin_path, meta)

    assert ds.n_train + ds.n_val == meta.n_tokens
    assert ds.n_val == pytest.approx(0.2 * meta.n_tokens, rel=0.01)

    x, y = ds.get_batch("train", batch_size=4, block_size=8, device=torch.device("cpu"))
    assert x.shape == y.shape == (4, 8)
    # targets are inputs shifted by one position
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert int(x.max()) < tok.vocab_size


def test_dataset_errors_on_split_too_small(tmp_path):
    meta = write_tokens(tmp_path / "t.bin", list(range(20)), vocab_size=50, level="char")
    ds = TokenDataset(tmp_path / "t.bin", meta)
    with pytest.raises(ValueError, match="block_size"):
        ds.get_batch("train", batch_size=2, block_size=64, device=torch.device("cpu"))


def test_synthetic_corpus_is_deterministic_and_structured():
    a = generate_corpus(target_chars=2000, seed=3)
    b = generate_corpus(target_chars=2000, seed=3)
    assert a == b
    assert len(a) >= 2000
    assert a.count(".") > 10  # sentence-ish structure, not noise
    assert a != generate_corpus(target_chars=2000, seed=4)


def test_memmap_reads_back_exact_values(tmp_path):
    ids = np.arange(500) % 37
    meta = write_tokens(tmp_path / "m.bin", ids, vocab_size=37, level="char")
    arr = np.memmap(tmp_path / "m.bin", dtype=np.dtype(meta.dtype), mode="r", shape=(meta.n_tokens,))
    assert np.array_equal(np.asarray(arr), ids.astype(np.dtype(meta.dtype)))
