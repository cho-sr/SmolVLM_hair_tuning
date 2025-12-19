# SmolVLM Hair Diagnosis & Recommendation (QLoRA Fine-tuning)

초경량 멀티모달 모델 **SmolVLM-256M-Instruct**를 **QLoRA**로 파인튜닝하여  
사용자 모발/두피 이미지 기반으로 **헤어 상태 진단 + 맞춤 케어/스타일 추천**을 생성하는 프로젝트입니다.

> ✅ **모델 가중치(체크포인트)는 GitHub에 포함하지 않습니다.**  
> 용량 제한 때문에 가중치는 Hugging Face에서 확인/다운로드 해주세요:  
> **https://huggingface.co/cho-sr/crying_cv/**

---

##  Features
- 이미지(모발/두피) 입력 → 텍스트 기반 **진단/추천 생성**
- **QLoRA** 기반 효율적 튜닝(저자원 환경 고려)
- (선택) **API 서버** 형태로 추론 제공 가능

---

##  Model
- Base: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Fine-tuned Weights: `cho-sr/crying_cv` (Hugging Face)
- Training: QLoRA (PEFT/LoRA)

