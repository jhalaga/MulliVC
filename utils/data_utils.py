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
        self.split = split
        self.max_samples = max_samples
        
        # Load datasets
        self.libritts_data = self._load_libritts()
        self.fongbe_data = self._load_fongbe()
        
        # Create sample lists
        self.samples = self._create_samples()
        
        if max_samples:
            self.samples = self.samples[:max_samples]

    def _split_name(self, train_split: str, eval_split: str) -> str:
        """Maps repository split names onto dataset split names."""
        if self.split in {'validation', 'val', 'dev', 'test'}:
            return eval_split
        return train_split

    def _sample_limit(self, default_limit: int, num_sources: int = 1) -> int:
        """Calculates a small per-source limit when max_samples is requested."""
        if not self.max_samples:
            return default_limit

        per_source_limit = max(1, self.max_samples // max(1, num_sources))
        return min(default_limit, per_source_limit)

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
                raise ValueError('Aucun contenu audio décodable trouvé dans l\'échantillon.')

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
    
    def _load_libritts(self) -> List[Dict]:
        """Loads the LibriTTS dataset."""
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "mythicinfinity/libritts",
                "clean",
                split=self._split_name("train.clean.360", "dev.clean"),
                streaming=True
            )
            dataset = dataset.cast_column('audio', Audio(decode=False))
            
            # Convert to a list for easier access
            libritts_samples = []
            sample_limit = self._sample_limit(1000)
            for i, sample in enumerate(dataset):
                if i >= sample_limit:
                    break
                libritts_samples.append({
                    'audio': self._decode_audio(
                        sample['audio'],
                        self.config['data']['sample_rate']
                    ),
                    'text': sample['text_normalized'],
                    'speaker_id': sample['speaker_id'],
                    'language': 'en'
                })
            
            return libritts_samples
            
        except Exception as e:
            print(f"Erreur lors du chargement de LibriTTS: {e}")
            return []
    
    def _load_fongbe(self) -> List[Dict]:
        """Loads the Fongbe dataset."""
        try:
            fongbe_samples = []
            sample_limit = self._sample_limit(500, num_sources=2)

            for config_name in ['female', 'male']:
                dataset = load_dataset(
                    "beethogedeon/fongbe-speech",
                    config_name,
                    split=self._split_name('train', 'test'),
                    streaming=True
                )
                dataset = dataset.cast_column('audio', Audio(decode=False))

                for i, sample in enumerate(dataset):
                    if i >= sample_limit:
                        break
                    fongbe_samples.append({
                        'audio': self._decode_audio(
                            sample['audio'],
                            self.config['data']['sample_rate']
                        ),
                        'text': sample['text'],
                        'speaker_id': str(sample['speaker_id']),
                        'language': 'fongbe'
                    })
            
            return fongbe_samples
            
        except Exception as e:
            print(f"Erreur lors du chargement de Fongbe: {e}")
            return []
    
    def _create_samples(self) -> List[Dict]:
        """Creates the sample list for training."""
        samples = []
        
        # Add LibriTTS samples
        for sample in self.libritts_data:
            samples.append({
                'audio': sample['audio'],
                'text': sample['text'],
                'speaker_id': sample['speaker_id'],
                'language': sample['language'],
                'dataset': 'libritts'
            })
        
        # Add Fongbe samples
        for sample in self.fongbe_data:
            samples.append({
                'audio': sample['audio'],
                'text': sample['text'],
                'speaker_id': sample['speaker_id'],
                'language': sample['language'],
                'dataset': 'fongbe'
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Retrieves a sample from the dataset."""
        sample = self.samples[idx]
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
        for sample in self.samples:
            if (sample['speaker_id'] == speaker_id and 
                sample['language'] == language):
                speaker_samples.append(sample)
        return speaker_samples
    
    def get_random_speaker(self, language: str) -> str:
        """Retrieves a random speaker ID for a given language."""
        language_samples = [s for s in self.samples if s['language'] == language]
        if not language_samples:
            return None
        
        speaker_ids = list(set([s['speaker_id'] for s in language_samples]))
        return random.choice(speaker_ids)


def create_dataloader(
    config: dict,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Creates a DataLoader for MulliVC."""
    
    dataset = MulliVCDataset(config, split=split)
    
    batch_size = batch_size or config['data']['batch_size']
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config['data']['pin_memory'],
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
