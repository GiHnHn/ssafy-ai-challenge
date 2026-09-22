# 실행 안내

## 최종 앙상블

보관된 예측 확률과 검색 후보가 있으면 GPU 없이 실행할 수 있습니다.

```sh
pip install -r requirements.txt
python ensemble.py --data-root /path/to/local-artifacts --output outputs/submission.csv
```

```text
local-artifacts/
  성능파일/
    qwen3_vl_8b_test.csv
    qwen38_adapter_counting_test.csv
    qwen35_tuned_test.csv
  640_file/qwen35_640_epoch2_test.csv
  TTA_file/qwen35_tta_original768_test.csv
  full_refit_file/
    qwen35_full_refit_640_test.csv
    qwen35_full_refit_768_test.csv
  near_duplicate_transfer_test_d0.csv
```

확률 CSV에는 `id`, `prob_a`, `prob_b`, `prob_c`, `prob_d`가 필요합니다. 8B 파일은 `category`를 포함하며, 27B 파일은 counting 문항만 담습니다. 검색 후보에는 `id`, `retrieval_pred`, `similarity`, `image_distance`가 필요합니다.

`retrieval` 스크립트는 `SSAFY_DATA_ROOT` 또는 저장소의 `data` 폴더를 사용합니다. train/dev/test CSV·이미지와 기준 예측을 준비합니다.

## 학습 자료

`notebooks`에는 27B counting, 9B의 512·640px 학습, 768px 추론, 전체 데이터 재학습 코드가 있습니다. Colab/Drive와 GPU를 전제로 하므로 설치 셀과 데이터 경로를 확인하세요.

`archive`는 과거 분석 원본입니다. 8B 노트북은 최종 체크포인트와 설정이 완전히 같다고 확인되지 않았습니다. `archive/ensemble`은 원래 파일 구조가 필요하고 import 시 계산되므로, 최종 결합에는 루트의 `ensemble.py`를 사용합니다.

## 검증 범위

`ensemble.py`는 공개 정리 시 최종 수식을 분리한 코드입니다. 보관된 입력으로 생성한 CSV가 당시 최종 제출 5,074개 ID·답안과 일치했습니다. GPU 재학습과 대회 서버 점수 재측정은 수행하지 않았습니다.

전체 데이터 재학습에는 과거 검증셋도 포함되어, 이후 같은 검증셋 점수를 독립 평가 성능으로 해석할 수 없습니다.
