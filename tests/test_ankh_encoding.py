"""Offline integration tests: actual T5 encoder with a small residue tokenizer."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, T5Config, T5EncoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from plm.sequence_length import resolve_token_length
from encoding.plm_encoding import compute_embeddings, main, get_parser, tokenize_sequences


class AnkhEncodingTest(unittest.TestCase):
    def setUp(self):
        backend = Tokenizer(WordLevel({'<pad>': 0, '</s>': 1, '<unk>': 2, 'A': 3, 'C': 4, 'D': 5}, unk_token='<unk>'))
        backend.post_processor = TemplateProcessing(single='$A </s>', special_tokens=[('</s>', 1)])
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='</s>', unk_token='<unk>')
        torch.manual_seed(0)
        self.model = T5EncoderModel(T5Config(vocab_size=6, d_model=8, d_ff=16, num_layers=1, num_heads=2, d_kv=4, dropout_rate=0)).eval()

    def test_residue_tokenization_pooling_and_batch_invariance(self):
        tokens = tokenize_sequences(['AC', 'D'], 6, self.tokenizer, is_ankh=True)
        self.assertEqual(tokens['input_ids'][0].tolist(), [3, 4, 1, 0, 0, 0])
        truncated = tokenize_sequences(['ACDACD'], 4, self.tokenizer, is_ankh=True)
        self.assertEqual(truncated['input_ids'][0].tolist(), [3, 4, 5, 1])
        kwargs = dict(model=self.model, sequences=['AC', 'D'], tokenizer=self.tokenizer, device=torch.device('cpu'), max_length=6, is_ankh=True)
        batched = compute_embeddings(**kwargs, batch_size=2)
        individual = compute_embeddings(**kwargs, batch_size=1)
        torch.testing.assert_close(batched, individual)
        with torch.no_grad():
            hidden = self.model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask']).last_hidden_state
        torch.testing.assert_close(batched[0], hidden[0, :2].mean(0))
        self.assertTrue(torch.isfinite(batched).all())
        self.assertEqual(batched.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, 'no residue'):
            compute_embeddings(**{**kwargs, 'sequences': ['']})

    def test_automatic_length_and_safety_cap(self):
        self.assertEqual(resolve_token_length(['A' * 300, 'AC'], self.tokenizer), 513)
        limit = resolve_token_length(['A' * 2000], self.tokenizer)
        self.assertEqual(limit, 513)
        tokens = tokenize_sequences(['A' * 2000, 'AC'], limit, self.tokenizer, is_ankh=True)
        self.assertEqual(tokens['input_ids'].shape, (2, 513))
        self.assertEqual(tokens['input_ids'][0, -1].item(), self.tokenizer.eos_token_id)
        self.assertEqual((tokens['input_ids'][0] == 3).sum().item(), 512)
        self.assertEqual(resolve_token_length(['A' * 2000], self.tokenizer, 1024), 1025)
        self.assertEqual(resolve_token_length(['A' * 300], self.tokenizer, 100), 101)
        self.tokenizer.model_max_length = 200
        self.assertEqual(resolve_token_length(['A' * 300], self.tokenizer), 200)
        for sequences, ceiling in [([], None), ([''], None), (['AC'], 0)]:
            with self.assertRaises(ValueError):
                resolve_token_length(sequences, self.tokenizer, ceiling)

    def test_power_of_two_boundaries(self):
        for residues, budget in [(1, 1), (2, 2), (3, 4), (82, 128), (147, 256),
                                 (255, 256), (256, 256), (257, 512), (512, 512), (513, 512)]:
            with self.subTest(residues=residues):
                self.assertEqual(resolve_token_length(['A' * residues], self.tokenizer), budget + 1)
        self.assertEqual(resolve_token_length(['A' * 257], self.tokenizer, 300), 301)
        for ceiling in [-1, 0]:
            with self.assertRaises(ValueError):
                resolve_token_length(['AC'], self.tokenizer, ceiling)

    def test_two_boundary_tokens_preserve_same_residue_budget(self):
        from types import SimpleNamespace
        esm_tokenizer = SimpleNamespace(num_special_tokens_to_add=lambda pair: 2)
        self.assertEqual(resolve_token_length(['A' * 2000], esm_tokenizer), 514)
        self.assertEqual(resolve_token_length(['AC'], esm_tokenizer), 4)

    def test_hdf5_alignment_and_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            query = Path(tmp) / 'query.csv'
            query.write_text('header,sequence,label\nb,AC,second\na,D,first\n')
            out = Path(tmp) / 'embeddings.h5'
            config = vars(get_parser().parse_args(['--model', 'ElnaggarLab/ankh-base', '--query', str(query), '--output', str(out)]))
            with patch('encoding.plm_encoding.AutoTokenizer.from_pretrained', return_value=self.tokenizer), patch('encoding.plm_encoding.T5EncoderModel.from_pretrained', return_value=self.model) as loader, patch('torch.cuda.is_available', return_value=False):
                main(config)
            loader.assert_called_once_with('ElnaggarLab/ankh-base')
            with h5py.File(out) as f:
                self.assertEqual(f['headers'][:].tolist(), [b'b', b'a'])
                self.assertEqual(f['labels'][:].tolist(), [b'second', b'first'])
                self.assertEqual(f['embeddings'].shape, (2, 8))
                self.assertEqual(f['embeddings'].dtype, np.dtype('float32'))
                self.assertEqual(f.attrs['model_id'], 'ElnaggarLab/ankh-base')
                self.assertEqual(f.attrs['max_length_tokens'], 3)
                self.assertEqual(f.attrs['max_length_residues'], 2)


if __name__ == '__main__':
    unittest.main()
