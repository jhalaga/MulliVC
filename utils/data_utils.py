"""
Utilities for data loading and processing.
"""
import io
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import random
from datasets import Audio, load_dataset
import soundfile as sf
import yaml


def _normalize_config_values(value):
    """Recursively converts numeric-looking YAML strings into Python numbers."""
    if isinstance(value, dict):
        return {key: _normalize_config_values(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_normalize_config_values(item) for item in value]

    if isinstance(value, str):
        stripped = value.strip()
        try:
            if any(marker in stripped.lower() for marker in ['.', 'e']):
                return float(stripped)
            return int(stripped)
        except ValueError:
            return value

    return value


class MulliVCDataset(Dataset):
    """Dataset for MulliVC with multilingual support."""
    
    def __init__(
        self,
        config: dict,
        split: str = "train",
        max_samples: Optional[int] = None
    ):
        self.config = config
        self.data_config = config['data']
        self.split = split
        self.max_samples = max_samples
        self.sample_rate = self.data_config['sample_rate']
        self.use_streaming = self.data_config.get('use_streaming', True)
        self.fongbe_configs = self.data_config.get('fongbe_configs', ['female', 'male'])

        self.sources = self._load_sources()
        self.samples = self._build_samples()

    def _split_name(
        self,
        train_key: str,
        eval_key: str,
        default_train: str,
        default_eval: str
    ) -> str:
        """Maps repository split names onto dataset split names."""
        if self.split in {'validation', 'val', 'dev', 'test'}:
            return self.data_config.get(eval_key, default_eval)
        return self.data_config.get(train_key, default_train)

    def _default_max_samples(self) -> Optional[int]:
        """Returns the default per-split sample cap from config."""
        if self.split in {'validation', 'val', 'dev', 'test'}:
            limit = self.data_config.get('max_validation_samples', 256)
        else:
            limit = self.data_config.get('max_train_samples', 2000)

        if self.use_streaming and limit is None:
            return 256 if self.split in {'validation', 'val', 'dev', 'test'} else 2000

        return limit

    def _prefetch_limit(self, num_sources: int) -> Optional[int]:
        """Computes a per-source cap for streaming mode."""
        total_limit = self.max_samples if self.max_samples is not None else self._default_max_samples()
        if total_limit is None:
            return None

        return max(1, (int(total_limit) + max(1, num_sources) - 1) // max(1, num_sources))

    def _decode_audio(self, audio_info: Dict, sample_rate: int) -> torch.Tensor:
        """Decodes dataset audio payloads without requiring torchcodec."""
        if 'array' in audio_info:
            audio_tensor = torch.from_numpy(audio_info['array']).float()
            audio_sr = audio_info.get('sampling_rate', sample_rate)
        else:
            audio_bytes = audio_info.get('bytes')
            audio_path = audio_info.get('path')

            if audio_bytes is not None:
                audio_array, audio_sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
            elif audio_path is not None:
                audio_array, audio_sr = sf.read(audio_path, always_2d=False)
            else:
                raise ValueError('No decodable audio content found in the sample.')

            audio_tensor = torch.from_numpy(audio_array).float()

        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.mean(dim=-1)

        if audio_sr != sample_rate:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor.unsqueeze(0),
                audio_sr,
                sample_rate
            ).squeeze(0)

        return audio_tensor

    def _load_sources(self) -> List[Dict]:
        """Loads dataset sources according to the active split and mode."""
        source_specs = [
            {
                'repo_name': 'mythicinfinity/libritts',
                'config_name': self.data_config.get('libritts_config', 'clean'),
                'split_name': self._split_name(
                    'libritts_train_split',
                    'libritts_validation_split',
                    'train.clean.360',
                    'dev.clean'
                ),
                'dataset_label': 'libritts',
                'language': 'en',
                'text_key': 'text_normalized',
                'speaker_key': 'speaker_id'
            }
        ]

        for config_name in self.fongbe_configs:
            source_specs.append({
                'repo_name': 'beethogedeon/fongbe-speech',
                'config_name': config_name,
                'split_name': self._split_name(
                    'fongbe_train_split',
                    'fongbe_validation_split',
                    'train',
                    'test'
                ),
                'dataset_label': 'fongbe',
                'language': 'fongbe',
                'text_key': 'text',
                'speaker_key': 'speaker_id'
            })

        sample_limit = self._prefetch_limit(len(source_specs))
        sources = []
        for spec in source_specs:
            if self.use_streaming:
                sources.append(self._load_streaming_source(spec, sample_limit))
            else:
                sources.append(self._load_indexed_source(spec))

        return [source for source in sources if self._source_length(source) > 0]

    def _load_streaming_source(self, spec: Dict, sample_limit: Optional[int]) -> Dict:
        """Loads a bounded streaming subset into memory for local experimentation."""
        try:
            dataset = load_dataset(
                spec['repo_name'],
                spec['config_name'],
                split=spec['split_name'],
                streaming=True
            )
            dataset = dataset.cast_column('audio', Audio(decode=False))

            samples = []
            for index, sample in enumerate(dataset):
                if sample_limit is not None and index >= sample_limit:
                    break
                samples.append({
                    'audio': self._decode_audio(sample['audio'], self.sample_rate),
                    'text': sample[spec['text_key']],
                    'speaker_id': str(sample[spec['speaker_key']]),
                    'language': spec['language'],
                    'dataset': spec['dataset_label']
                })

            return {
                'kind': 'streaming',
                'samples': samples,
                **spec
            }
        except Exception as exc:
            print(
                f"Error loading {spec['repo_name']}"
                f"/{spec['config_name']} ({spec['split_name']}): {exc}"
            )
            return {
                'kind': 'streaming',
                'samples': [],
                **spec
            }

    def _load_indexed_source(self, spec: Dict) -> Dict:
        """Loads an indexable dataset for longer cloud training runs."""
        try:
            dataset = load_dataset(
                spec['repo_name'],
                spec['config_name'],
                split=spec['split_name'],
                streaming=False
            )
            dataset = dataset.cast_column('audio', Audio(decode=False))

            return {
                'kind': 'indexed',
                'dataset': dataset,
                **spec
            }
        except Exception as exc:
            print(
                f"Error loading {spec['repo_name']}"
                f"/{spec['config_name']} ({spec['split_name']}): {exc}"
            )
            return {
                'kind': 'indexed',
                'dataset': None,
                **spec
            }

    def _source_length(self, source: Dict) -> int:
        """Returns the number of available samples for a source."""
        if source['kind'] == 'streaming':
            return len(source['samples'])

        if source.get('dataset') is None:
            return 0

        return len(source['dataset'])

    def _build_samples(self) -> List[Dict]:
        """Builds an interleaved sample index across all sources."""
        if not self.sources:
            return []

        sample_refs = []
        total_limit = self.max_samples if self.max_samples is not None else self._default_max_samples()
        max_source_length = max(self._source_length(source) for source in self.sources)

        for item_index in range(max_source_length):
            for source_index, source in enumerate(self.sources):
                if item_index >= self._source_length(source):
                    continue

                if source['kind'] == 'streaming':
                    sample_refs.append(source['samples'][item_index])
                else:
                    sample_refs.append({
                        'source_index': source_index,
                        'item_index': item_index
                    })

                if total_limit is not None and len(sample_refs) >= total_limit:
                    return sample_refs

        return sample_refs

    def _load_indexed_sample(self, sample_ref: Dict) -> Dict:
        """Decodes one sample from an indexed source on demand."""
        source = self.sources[sample_ref['source_index']]
        raw_sample = source['dataset'][sample_ref['item_index']]

        return {
            'audio': self._decode_audio(raw_sample['audio'], self.sample_rate),
            'text': raw_sample[source['text_key']],
            'speaker_id': str(raw_sample[source['speaker_key']]),
            'language': source['language'],
            'dataset': source['dataset_label']
        }

    def _sample_metadata(self, sample_ref: Dict) -> Dict:
        """Returns lightweight metadata for speaker/language grouping."""
        if 'audio' in sample_ref:
            return {
                'text': sample_ref['text'],
                'speaker_id': sample_ref['speaker_id'],
                'language': sample_ref['language'],
                'dataset': sample_ref['dataset']
            }

        source = self.sources[sample_ref['source_index']]
        raw_sample = source['dataset'][sample_ref['item_index']]
        return {
            'text': raw_sample[source['text_key']],
            'speaker_id': str(raw_sample[source['speaker_key']]),
            'language': source['language'],
            'dataset': source['dataset_label']
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Retrieves a sample from the dataset."""
        sample = self.samples[idx]
        if 'audio' not in sample:
            sample = self._load_indexed_sample(sample)

        audio_tensor = sample['audio'].clone()
        
        # Normalize audio
        audio_tensor = audio_tensor / (torch.abs(audio_tensor).max() + 1e-8)
        
        return {
            'audio': audio_tensor,
            'text': sample['text'],
            'speaker_id': sample['speaker_id'],
            'language': sample['language'],
            'dataset': sample['dataset']
        }
    
    def get_speaker_samples(self, speaker_id: str, language: str) -> List[Dict]:
        """Retrieves all samples for a speaker in a given language."""
        speaker_samples = []
        for idx, sample in enumerate(self.samples):
            metadata = self._sample_metadata(sample)
            if (
                metadata['speaker_id'] == speaker_id and
                metadata['language'] == language
            ):
                speaker_samples.append(self[idx])
        return speaker_samples
    
    def get_random_speaker(self, language: str) -> str:
        """Retrieves a random speaker ID for a given language."""
        speaker_ids = {
            self._sample_metadata(sample)['speaker_id']
            for sample in self.samples
            if self._sample_metadata(sample)['language'] == language
        }
        if not speaker_ids:
            return None

        return random.choice(list(speaker_ids))


def create_dataloader(
    config: dict,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    max_samples: Optional[int] = None
) -> DataLoader:
    """Creates a DataLoader for MulliVC."""
    data_config = config['data']

    if max_samples is None:
        if split in {'validation', 'val', 'dev'}:
            max_samples = data_config.get('max_validation_samples')
        elif split == 'test':
            max_samples = data_config.get('max_test_samples')
        else:
            max_samples = data_config.get('max_train_samples')

    dataset = MulliVCDataset(config, split=split, max_samples=max_samples)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples were loaded for split '{split}'.")

    batch_size = batch_size or data_config['batch_size']
    if shuffle is None:
        shuffle = split == 'train'
    if num_workers is None:
        num_workers = data_config.get('num_workers', 0)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=data_config['pin_memory'],
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict]) -> Dict:
    """Collation function for the DataLoader."""
    # Split batch items
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]
    speaker_ids = [item['speaker_id'] for item in batch]
    languages = [item['language'] for item in batch]
    datasets = [item['dataset'] for item in batch]
    
    # Pad audio to the same length
    max_length = max([len(audio) for audio in audios])
    padded_audios = []
    
    for audio in audios:
        if len(audio) < max_length:
            padding = torch.zeros(max_length - len(audio))
            padded_audio = torch.cat([audio, padding])
        else:
            padded_audio = audio
        padded_audios.append(padded_audio)
    
    return {
        'audio': torch.stack(padded_audios),
        'text': texts,
        'speaker_id': speaker_ids,
        'language': languages,
        'dataset': datasets
    }


def create_cross_lingual_pairs(
    dataset: MulliVCDataset,
    num_pairs: int = 100
) -> List[Tuple[Dict, Dict]]:
    """
    Creates cross-lingual pairs for training.

    Args:
        dataset: MulliVC dataset.
        num_pairs: Number of pairs to create.

    Returns:
        List of tuples (sample_A, sample_B) where A and B are from different languages.
    """
    pairs = []
    
    # Get speakers by language
    en_speakers = list(set([
        s['speaker_id'] for s in dataset.samples 
        if s['language'] == 'en'
    ]))
    fongbe_speakers = list(set([
        s['speaker_id'] for s in dataset.samples 
        if s['language'] == 'fongbe'
    ]))
    
    for _ in range(num_pairs):
        # Choose an English speaker
        en_speaker = random.choice(en_speakers)
        en_samples = dataset.get_speaker_samples(en_speaker, 'en')
        if not en_samples:
            continue
        
        # Choose a Fongbe speaker
        fongbe_speaker = random.choice(fongbe_speakers)
        fongbe_samples = dataset.get_speaker_samples(fongbe_speaker, 'fongbe')
        if not fongbe_samples:
            continue
        
        # Create a pair
        en_sample = random.choice(en_samples)
        fongbe_sample = random.choice(fongbe_samples)
        
        pairs.append((en_sample, fongbe_sample))
    
    return pairs


def load_config(config_path: str) -> dict:
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return _normalize_config_values(config)
