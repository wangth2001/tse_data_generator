# Mixture and Augmentation Data Generator for TSE/ASV

用于目标说话人提取或说话人验证的混叠与增强数据生成器

## 数据准备

- 原始音频数据
  - 所有数据要求为 16 位 PCM-WAV 编码的单声道音频文件，可使用以下命令转换为该格式：
  - `ffmpeg -i "in.wav" -acodec pcm_s16le -ar 16000 -ac 1 "out.wav"`
- 数据列表
  - 输入的 clean 说话人数据列表格式为 `"spk" "audio_path"`，每行一条；
  - 各种噪声数据以及其他说话人的音频数据列表格式为 `"audio_path"`，每行一条；
  - 具体可参照 data_list 文件夹中的示例。
- 示例中的数据来源
  - 噪声、背景音乐与说话人声：[MUSAN](https://www.openslr.org/17/)
  - 混响：[Room Impulse Response and Noise Database](https://www.openslr.org/28/) 中的 simulated_rirs

## 使用说明

- 数据生成流程
  1. 根据参数，是/否对 clean 语音混叠其他说话人的语音。若混叠，则从 mix_speech 数据列表中随机选出 `mix_number` 条语音对 clean 语音进行混叠；
  2. 根据参数，是/否对上述处理后的语音进行数据增强。若增强，则从噪声、背景音乐与混响中随机挑选一种进行数据增强。
  3. 音频归一化，输出。
- 输出文件说明
  - 输出语音文件会按照说话人（clean 列表中的 `spk`）进行保存，每个说话人一个文件夹，因此同一个说话人的 clean 语音文件名不可以重复，否则在输出时会产生覆盖。
  - 输出文件名会在原文件名后加一些后缀以进行区分，包括该语音第几次生成（配置文件中的 `output_times`）、是否混叠与是否增强。

