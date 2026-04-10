"""
Utilities for data loading and processing.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
import random
from datasets import load_dataset
import yaml


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
    
    def _load_libritts(self) -> List[Dict]:
        """Loads the LibriTTS dataset."""
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "mythicinfinity/libritts",
                split="train.clean.360",
                streaming=True
            )
            
            # Convert to a list for easier access
            libritts_samples = []
            for i, sample in enumerate(dataset):
                if i >= 1000:  # Limit for tests
                    break
                libritts_samples.append({
                    'audio': sample['audio'],
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
            # Load from HuggingFace
            dataset = load_dataset(
                "beethogedeon/fongbe-speech",
                split="train",
                streaming=True
            )
            
            # Convert to a list
            fongbe_samples = []
            for i, sample in enumerate(dataset):
                if i >= 1000:  # Limit for tests
                    break
                fongbe_samples.append({
                    'audio': sample['audio'],
                    'text': sample['text'],
                    'speaker_id': sample['speaker_id'],
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
        
        # Load audio
        audio = sample['audio']['array']
        audio_tensor = torch.from_numpy(audio).float()
        
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
    return config
