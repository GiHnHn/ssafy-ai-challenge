# SSAFY AI Challenge · 이미지 질의응답

**이미지와 질문을 보고 정답을 선택하는 VQA 과제에서, 모델을 학습하고 예측을 결합해 성능을 높였습니다.**

PyTorch · Transformers · PEFT/LoRA · Qwen VLM

**Public 0.94994 · Private 5위**

보관된 제출 기록 및 본인 확인 기준

## 주요 접근

- VLM을 **LoRA로 학습**하고, 여러 해상도의 예측을 비교해 조합을 선택했습니다.
- 객체 수를 묻는 문항에는 counting 모델을 별도로 결합하고, 모델별 예측 확률을 가중 앙상블했습니다.
- 최종 모델 조합에 이미지 해시·질문 유사도 기반 후처리를 적용했습니다.

## 코드와 재현

| 코드 | 역할 |
|---|---|
| [notebooks](notebooks) | 모델 학습·추론 |
| [ensemble.py](ensemble.py) | 최종 예측 결합 |
| [retrieval](retrieval) | 중복 이미지 후처리 |

보관된 예측으로 **최종 제출 5,074개 답안 전체 일치**를 확인했습니다. GPU 재학습은 수행하지 않았습니다.

[실행 안내](docs/REPRODUCING.md) · 대회 데이터·예측 파일·모델 가중치는 별도 준비가 필요합니다.
