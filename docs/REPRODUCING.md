# 재현 안내

## 학습 노트북

| 공개 파일 | 원래 파일 | 역할 |
|---|---|---|
| `01_Qwen3_8_27B_Counting_LoRA` | `(260324)_baseline_colab.ipynb` | 파일명과 달리 보관 내용은 Qwen3.8-27B counting 실험 |
| `02_Qwen3_5_9B_512_VQA` | `Qwen3_5_9B_Full_VQA` | 512px 전체 VQA 학습 |
| `03_Qwen3_5_9B_640_VQA` | `Qwen3_5_9B_640_Full_VQA` | 640px fresh 학습, epoch2 선택 |
| `04_Qwen3_5_9B_768_Inference` | `Qwen3_5_9B_640_768_TTA` | 같은 adapter의 추가 해상도 추론 |
| `05_Qwen3_5_9B_FullData_Refit` | `Qwen3_5_9B_640_FullData_Refit` | 전체 데이터 재학습, 재개와 640/768 추론 |

8B 노트북은 백업에서 확보한 과거 학습 실험입니다. 최종 8B 기준 체크포인트와 완전히 동일한 학습 설정을 확정하는 파일로 소개하지 않습니다. 최종 8B/27B 예측 파일은 별도로 보관된 자료를 입력으로 사용합니다.

실행 출력·셀 실행 번호·개인 UI 메타데이터를 제거했습니다. 노트북 내부 패키지 설치 셀과 모델 ID는 당시 기록을 유지했으며, 현재 GPU 환경 전체에서 재학습하지 않았습니다. Drive 경로와 데이터 압축 파일 경로는 본인 환경에 맞춰 설정합니다.

## 최종 결합 입력 구조

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

확률 CSV는 `id`, `prob_a`, `prob_b`, `prob_c`, `prob_d` 열이 필요합니다. 8B 파일은 `category`도 필요하고, 27B 파일은 counting 문항만 포함합니다. 검색 후보에는 `id`, `retrieval_pred`, `similarity`, `image_distance`가 필요합니다. ID로 정렬하고 누락·중복·유효하지 않은 확률을 검사합니다.

```sh
python ensemble.py --data-root /path/to/local-artifacts --output outputs/submission.csv
```

`retrieval` 원본 스크립트는 `SSAFY_DATA_ROOT` 환경변수 또는 저장소의 `data` 폴더를 기준으로 동작합니다. train/dev/test CSV와 이미지, 기준 검증 예측이 필요합니다. 이미지 해시는 `analyze_image_duplicates.py`, 해시와 질문을 이용한 후보는 `analyze_near_duplicate_transfer.py`에서 계산합니다.

`archive/ensemble`은 당시 분석을 보존한 코드라 원래 폴더 구조·기존 제출 파일·검증 확률이 추가로 필요하고 import 시 계산이 시작됩니다. 간단한 최종 재현에는 루트의 `ensemble.py`를 사용합니다.

## 공개 정리 중 변경

최종 결합 수식만 별도 CLI로 분리하고 입력 ID 검사·경로 인자를 추가했습니다. 검색 후보와 기본 8B 답을 비교해 당시 후처리의 개입 범위를 보존했습니다. 학습 알고리즘을 새로 만든 것으로 설명하지 않습니다. 데이터·가중치·제출 CSV와 모델 출력은 공개본에서 제외했습니다.

## 공개 전 검증 (2026.09.22)

보관된 원본 예측 파일에 `ensemble.py`를 적용한 결과, 5,074개 ID와 답안이 당시 최종 제출 파일과 모두 일치했습니다. 중복 이미지 후처리의 개입은 3건이었으며, 생성 CSV의 바이트 단위 SHA-256도 보관본과 같았습니다.

```text
3ef45df9b1d67973a379e0b957c06111ccac9a600a8613537b0d9d59e47db300
```

이 검증은 최종 결합 코드가 기존 제출을 보존하는지 확인한 것입니다. GPU 재학습이나 대회 서버에서의 점수 재측정은 수행하지 않았습니다.
