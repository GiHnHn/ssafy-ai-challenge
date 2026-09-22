# SSAFY AI Challenge | VQA 모델 학습과 앙상블

**이미지와 질문을 보고 선택지 답을 예측하는 VQA 과제에서, 모델 학습·해상도·문항 유형별 확률 결합을 비교한 기록입니다.** 큰 모델을 하나 사용하는 접근에서 출발해, 각 모델의 예측을 결합하고 성능이 떨어진 실험을 제외하며 최종 제출을 구성했습니다.

PyTorch · Transformers · PEFT/LoRA · Qwen VLM · NumPy · pandas

| 결과 | 기록 기준 |
|---|---|
| Public 최고 | **0.94994** · 당시 제출·실험 기록 |
| Private 순위 | **5위** · 2026.09.12 본인 확인 기록 |
| 학습 장비 | NVIDIA A100-SXM4-80GB · bf16 |

수치와 순위는 보관 기록 기준이며 별도의 수상명을 주장하지 않습니다.

## 진행한 작업

- 기준 8B 모델과 27B counting 모델을 비교하고 문항 유형에 따라 결합했습니다.
- Qwen3.5-9B를 512·640 해상도에서 각각 학습하고 epoch별 검증 결과를 비교했습니다.
- 같은 adapter의 768 해상도 추론과 추가 전체 데이터 학습 결과를 조합했습니다.
- 객체 crop·SoM·flip 등 성능이 떨어진 실험도 기록해 최종 채택 항목과 구분했습니다.
- 확률 결합, ID 정렬, 중복 이미지 기반 후처리와 제출 파일의 계보를 관리했습니다.

## 최종 제출 구성

`GeoBlend`는 정규화한 선택지 확률의 가중 기하평균입니다.

```text
current = 8B
counting 문항만 current = GeoBlend(8B, 27B; 0.51, 0.49)
equal   = GeoBlend(current, 9B 512px; 0.50, 0.50)
anchor1 = GeoBlend(equal, 9B 640px epoch2; 0.50, 0.50)
anchor2 = GeoBlend(anchor1, 같은 adapter의 768px 추론; 0.825, 0.175)
refit   = GeoBlend(전체 데이터 재학습 640px, 768px; 0.825, 0.175)
final   = GeoBlend(anchor2, refit; 0.877, 0.123)
```

마지막에 train/dev 이미지·질문을 참조한 중복 이미지 후처리를 적용했습니다. 최종 코드에는 test ID별 정답을 직접 넣지 않으며, 질문 유사도와 이미지 해시로 정한 개입을 기본 8B 예측과 비교해 전달합니다. [실험과 평가](docs/EXPERIMENTS.md)에 채택·제외한 시도를 구분했습니다.

## 파일 안내

| 경로 | 역할 |
|---|---|
| [notebooks](notebooks) | 최종 구성에 연결되는 27B counting 및 9B 학습·추론·전체 재학습 노트북 |
| [ensemble.py](ensemble.py) | 최종 채택한 결합만 남긴 공개 정리용 실행 코드 |
| [retrieval](retrieval) | 이미지 해시·질문 유사도 기반 후보 계산과 후처리 원본 코드 |
| [archive](archive) | 기존 8B 학습 실험과 당시 앙상블 분석 코드 보존 |
| [재현 안내](docs/REPRODUCING.md) | 실행 순서·입력 파일·모델 및 데이터 준비 범위 |

## 실행

예측 확률 파일과 검색 후보가 준비되어 있다면 GPU 없이 최종 결합을 실행할 수 있습니다.

```sh
pip install -r requirements.txt
python ensemble.py --data-root /path/to/local-artifacts --output outputs/submission.csv
```

학습 노트북은 Colab/Drive와 GPU를 전제로 합니다. 데이터 경로와 모델 ID를 확인한 뒤 실행합니다. 저장소에는 **대회 원본 데이터, 예측 CSV, 학습 가중치, 노트북 실행 출력, 인증정보를 포함하지 않습니다.**

## 재현과 평가의 범위

- 보관된 예측 파일로 최종 결합을 실행해 **5,074개 ID·답안 전체가 당시 최종 제출과 일치**함을 확인했습니다. 생성한 CSV의 SHA-256도 동일합니다. GPU 학습은 새로 실행하지 않았습니다.
- 전체 재학습에는 과거 검증 데이터도 포함됐으므로, 이후 같은 검증 점수를 독립 평가 성능으로 해석하지 않습니다.
- `ensemble.py`는 2026년 공개 정리 과정에서 당시 최종 수식과 후처리를 분리한 코드입니다. 모델 성능을 새로 개선한 작업은 아닙니다.
- 노트북 파일명과 원래 파일의 대응 및 8B 체크포인트의 재현 제약은 [재현 안내](docs/REPRODUCING.md)를 참고하세요.
