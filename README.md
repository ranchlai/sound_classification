# Sound classification

This example demonstrates the use of pretrained models trained on Audioset to fine-tune a smaller sound classification dataset, namely esc50 dataset.

The tricks and tips are as follows:
- Use pre-trained model Cnn14 from <a href="https://github.com/qiuqiangkong/audioset_tagging_cnn">audioset_tagging_cnn</a>[1]. The model is pre-trained on audioset, which is the largest weakly-labelled(only class info, without exact event time location) sound event dataset.
- No weight-decaying is used
- In training, spectrogram is of 384 frames. Random cropping is used.
- In test/eval, all 501 frames are used (so the result is also deterministic.
- Same lr decreasing scheduler as in baseline.
- Use large dropout,

## Testing
run the following to test on esc-50 dataset. You might need
``` bash
python test.py -a <audio_folder> -m <meta_file> -d gpu
```

### Results
Without any tricks, this example achieved average acc 0.937 across 5 folds, ranking <b> No. 2 </b> in the leader board.

## Training
TBD
### Requirements
```bash
# install paddleaudio
git clone https://github.com/PaddlePaddle/models.git
cd models/PaddleAudio
pip install -e .
```

```bash
git clone https://github.com/ranchlai/sound_classification.git
cd sound_classification
pip install -r requirements.txt
```



## References
[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

[2] Urban Sound Tagging using Multi-Channel Audio Feature with Convolutional Neural Networks
