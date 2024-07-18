import os
import fire
import yaml
import random
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile


class DataProcessor:
    def __init__(self, sample_rate):
        assert isinstance(sample_rate, int)
        self.sr = sample_rate
    
    def load_wav(self, audio_path, resample=True, norm=True):
        sr, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32)
        if resample and sr != self.sr:
            audio = signal.resample(audio, int(audio.shape[0] / sr * self.sr))
            sr = self.sr
        if norm:
            audio = audio / (1 << 15)
        return audio, sr
    
    def norm_wav(self, wav):
        #  norm wav value to [-1.0, 1.0]
        wav = wav / (np.max(np.abs(wav)) + 1e-4)
        return wav

    def random_chunk(self, audio, chunk_len):
        data_len = audio.shape[0]
        # random chunk
        if data_len >= chunk_len:
            chunk_start = random.randint(0, data_len - chunk_len)
            audio = audio[chunk_start:chunk_start + chunk_len]
        else:
            # padding
            repeat_times = chunk_len // data_len + 1
            audio = np.tile(audio, repeat_times)[:chunk_len]
        return audio
    
    def sample_snr(self, configs):
        assert configs['snr_sampling'] in ('uniform', 'gaussian')
        if configs['snr_sampling'] == 'uniform':
            snr = random.uniform(configs['snr_min'], configs['snr_max'])
        else:
            snr = random.gauss(configs['snr_mean'], configs['snr_var'] ** 0.5)
        return snr
    
    def snr_scale(self, ori_audio, add_audio, snr):
        ori_audio_db = 10 * np.log10(np.mean(ori_audio ** 2) + 1e-4)
        add_audio_db = 10 * np.log10(np.mean(add_audio ** 2) + 1e-4)
        add_audio_scale = np.sqrt(10**((ori_audio_db - add_audio_db - snr) / 10)) * add_audio
        return add_audio_scale
    
    def mix_speech(self, target_speech, mix_speech_list, configs):
        speech_len = target_speech.shape[0]
        mix_number = configs['mix_number']
        interfer_speeches = []
        for i in range(mix_number):
            interfer_speech, _ = self.load_wav(random.choice(mix_speech_list), resample=True, norm=True)
            interfer_speech = self.random_chunk(interfer_speech, chunk_len=speech_len)
            interfer_speech = self.snr_scale(target_speech, interfer_speech, self.sample_snr(configs))
            interfer_speeches.append(interfer_speech)
        out_speech = target_speech + np.sum(interfer_speeches, axis=0)
        return out_speech
    
    def additive_noise(self, audio, noise_list, configs):
        audio_len = audio.shape[0]
        noise_audio, _ = self.load_wav(random.choice(noise_list), resample=True, norm=True)
        noise_audio = self.random_chunk(noise_audio, chunk_len=audio_len)
        noise_audio = self.snr_scale(audio, noise_audio, self.sample_snr(configs))
        out_audio = audio + noise_audio
        return out_audio
    
    def reverberate(self, audio, reverb_list):
        audio_len = audio.shape[0]
        rir_audio, _ = self.load_wav(random.choice(reverb_list), resample=True, norm=False)
        rir_audio = rir_audio.astype(np.float32)
        rir_audio = rir_audio / np.sqrt(np.sum(rir_audio ** 2))
        out_audio = signal.convolve(audio, rir_audio, mode='full')[:audio_len]
        return out_audio


def read_list(data_list_path):
    lst = []
    with open(data_list_path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 1:
                line = line[0]
            lst.append(line)
    return lst


def main(config='config.yaml', **kwargs):
    # 加载配置文件
    with open(config) as conf:
        yaml_config = yaml.load(conf, Loader=yaml.FullLoader)
    configs = dict(yaml_config, **kwargs)
    sample_rate = configs['sample_rate']
    processor = DataProcessor(sample_rate=sample_rate)
    # print(f'configs: {configs}')
    
    # 读取数据列表
    input_list = read_list(configs['input_data_list_path'])
    print(f'input data number: {len(input_list)}')
    mix_speech_flag = configs['mix_speech']
    augmentation_flag = configs['augmentation']
    noise_flag = augmentation_flag and configs['augmentation_args']['noise']
    music_flag = augmentation_flag and configs['augmentation_args']['music']
    reverb_flag = augmentation_flag and configs['augmentation_args']['reverb']
    if mix_speech_flag:
        mix_speech_list = read_list(configs['mix_speech_args']['mix_speech_list_path'])
        print(f'mix speech data number: {len(mix_speech_list)}')
    if augmentation_flag:
        aug_methods = []
        if noise_flag:
            aug_methods.append('noise')
            noise_list = read_list(configs['augmentation_args']['noise_args']['noise_list_path'])
            print(f'noise data number: {len(noise_list)}')
        if music_flag:
            aug_methods.append('music')
            music_list = read_list(configs['augmentation_args']['music_args']['music_list_path'])
            print(f'music data number: {len(music_list)}')
        if reverb_flag:
            aug_methods.append('reverb')
            reverb_list = read_list(configs['augmentation_args']['reverb_args']['reverb_list_path'])
            print(f'reverberation data number: {len(reverb_list)}')
        augmentation_flag = noise_flag or music_flag or reverb_flag
    
    # 生成数据
    output_list = []
    output_times = configs['output_times']
    for spk, audio_path in tqdm(input_list):
        spk_output_path = os.path.join(configs['output_data_path'], spk)
        os.makedirs(spk_output_path, exist_ok=True)
        audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]
        audio, sr = processor.load_wav(audio_path, resample=True, norm=True)
        for i in range(output_times):
            out_audio = audio.copy()
            suffix = f'_time{i + 1}'
            if mix_speech_flag:
                out_audio = processor.mix_speech(out_audio, mix_speech_list, configs['mix_speech_args'])
                suffix += '_mix'
            if augmentation_flag:
                aug_type = random.choice(aug_methods)
                if aug_type == 'noise':
                    out_audio = processor.additive_noise(out_audio, noise_list, configs['augmentation_args']['noise_args'])
                elif aug_type == 'music':
                    out_audio = processor.additive_noise(out_audio, music_list, configs['augmentation_args']['music_args'])
                else:
                    out_audio = processor.reverberate(out_audio, reverb_list)
                suffix += f'_{aug_type}'
            out_audio = processor.norm_wav(out_audio)
            save_path = os.path.join(spk_output_path, audio_file_name + suffix + '.wav')
            output_list.append([spk, save_path])
            sf.write(save_path, out_audio, sample_rate)
    
    with open(configs['output_data_list_path'], 'w') as f:
        for spk, audio_path in output_list:
            f.write(f'{spk} {audio_path}\n')
    print(f'generate done, output number: {len(output_list)}')

if __name__ == '__main__':
    fire.Fire(main)
