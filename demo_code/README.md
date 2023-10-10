# Demo code

Demo code를 작동하기 위해서 크게 2step으로 나뉩니다. 먼저 데모에 활용할 비디오들을 전처리 합니다. 전처리가 완료되면 비디오를 선택하여 데모 영상을 만듭니다.

Demo에 사용하기 적합한 비디오는 다음 조건을 만족해야 합니다.
- 음성이 있는 동영상
- 화면 전환이 없는 동영상
- 한 사람이 등장하는 동영상

### 데이터 전처리
1. 데이터 전처리할 동영상을 '../assets/demo_videos' 폴더 안에 저장합니다.
2. 다음 명령어를 통해 '../assets/demo_videos' 안에 있는 동영상 전체에 대해서 전처리 합니다.
    ```python
    python data_pp.py
    ```
3. 다음 명령어를 통해 demo 동영상을 만듭니다.
    ```python
    python run_main.py
    ```
    - run_main.py의 'driving_clip_names', 'source_clip_names' list를 수정해서 demo에 사용하는 동영상을 선택합니다.
    - driving_clip_names : lipsync할 동영상 목록 입니다.
    - source_clip_names : 따라할 음성 목록 입니다.
    - demo video로 source video의 음성을 따라한 driving video가 만들어 집니다.
