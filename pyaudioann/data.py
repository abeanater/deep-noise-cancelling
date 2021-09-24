from pydub import AudioSegment
from pydub import effects
from scipy.io import wavfile
import os as os
from lxml import etree
import xml.dom.minidom
from lxml.html import parse
import logging
from torch.utils.data import Dataset
import pandas as pd

LOG = logging.getLogger(__file__)


def get_data_dir():
    package_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(package_dir, os.pardir, 'data'))


def generate_active_noise_out():
    """Generate datasets for active noise cancelation.  We want the noise to be inverse since
    we plan on canceling it out.
    """
    LOG.info("Generating desired signals")

    OUT_DIR = os.path.join(get_data_dir(), "InverseNoise")
    IN_DIR = os.path.join(get_data_dir(), "Nonspeech")

    # get all feature source code
    for root, dirs, files in os.walk(IN_DIR):
        for f in files:
            LOG.info(root + f)
            LOG.info("Noise File:         " + root + f)
            LOG.info("Inverse Noise File: " + OUT_DIR + f)
            out_rate, out_data = wavfile.read(root + f)
            out_data *= -1
            wavfile.write(OUT_DIR + f, out_rate, out_data)


def generate_datasets():
    ''' Generating datasets for noise filtering
    '''
    LOG.info("Reading in noise")
    samples = []
    non_speech_folder = os.path.join(get_data_dir(), "Nonspeech")
    clean_speech_folder = os.path.join(get_data_dir(), "Speech")
    out_folder = os.path.join(get_data_dir(), "labeled_datasets")
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    x = 0
    # get all feature source code
    for root, dirs, files in os.walk(non_speech_folder):
        for f in files:
            LOG.info(root + f)
            for rootSpeech, dirsSpeech, filesSpeech in os.walk(clean_speech_folder):
                for f2 in filesSpeech:
                    sample = {}
                    sample['noise_file'] = os.path.join(root,f)
                    sample['speech_file'] = os.path.join(rootSpeech,f2)
                    sample['overlayed_file'] = os.path.join(out_folder, str(x) + ".wav")

                    LOG.info("Noise File: " + sample['noise_file'])
                    LOG.info("Speech File: " +  sample['speech_file'])
                    noise = AudioSegment.from_file(sample['noise_file'])
                    speech = AudioSegment.from_file(sample['speech_file'])
                    combined = speech.overlay(noise, loop=True)
                    combined.export(sample['overlayed_file'], format = 'wav')
                    samples.append(sample)
                    x +=1 
    pd.DataFrame(samples).to_json(os.path.join(out_folder,'labels.json'))


class AudioDataset(Dataset):
    def __init__(self, annotations_file, cfg, transform = None, target_transform = None):
        parser=etree.XMLParser(remove_blank_text = True)
        self.labels=pd.read_json(annotations_file)
        self.transform=transform
        self.target_transform=target_transform
        self.cfg = cfg

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            sample = self.labels.iloc[idx,:].to_dict()
            
            
            x = AudioSegment.from_file(sample['overlayed_file'], format="wav").get_array_of_samples()
            if self.cfg['model_predict_speech']:
                y = AudioSegment.from_file(sample['speech_file'], format="wav").get_array_of_samples()
            else:
                y = AudioSegment.from_file(sample['noise_file'], format="wav").get_array_of_samples()
            
            if self.transform:
                x = self.transform(x)
            
            if self.target_transform:
                y = self.target_transform(y)
            
            return x,y