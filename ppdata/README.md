# ppdata

lipsync 모델을 학습하기 위한 dataset을 전처리하기 위한 코드입니다. 전처리가 완료되면 데이터를 모아두는 폴더에 옮기고 학습 과정을 진행하시면 됩니다.

lipsync 모델 학습에 사용하기 적합한 비디오는 다음 조건을 만족해야 합니다.
- 음성이 있는 동영상
- 화면 전환이 없는 동영상
- 한 사람이 등장하는 동영상

### 데이터 전처리
- 기존 데이터셋은 {dataset_root_folder}/{id_number}/{video_name}/{video_number}/{data} 방식의 폴더 트리를 가지고 있습니다.
- 추가되는 데이터셋은 id_number를 99999 숫자부터 역순으로 채워나가고 있습니다.
- 데이터를 모아둔 폴더에서 가장 최근에 추가된 숫자를 파악하고, 해당 숫자의 -1 값으로 gen_dataset.py 파일의 id_num 설정을 바꿔주셔야 합니다.


```python
python gen_dataset.py
```
