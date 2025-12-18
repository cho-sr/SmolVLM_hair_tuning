"""
SmolVLM Webcam Auto Inference (Fine-tuned)
3초마다 자동으로 inference 수행
Fine-tuned on Hair classification & description dataset
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import gradio as gr
import numpy as np
from datetime import datetime
import time

# =========================
# 설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
FINETUNED_MODEL_PATH = "/root/crying_cv_vlm/checkpoint-105"  # ✅ 최종 학습된 모델 (checkpoint-105)
INFERENCE_INTERVAL = 3  # 3초 간격

print(f"🔧 Device: {DEVICE}")
print(f"📂 Fine-tuned Model: {FINETUNED_MODEL_PATH}")
print("Loading model...")

# =========================
# 모델 로드 (Fine-tuned LoRA)
# =========================
from transformers import AutoModelForImageTextToText
from peft import PeftModel

print("1️⃣ Loading base model...")
model = AutoModelForImageTextToText.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",
    attn_implementation="eager"
)

print("2️⃣ Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(
    model,
    FINETUNED_MODEL_PATH,
    device_map="auto"
)

print("3️⃣ Merging adapter...")
model = model.merge_and_unload()
model.eval()

print("4️⃣ Loading processor...")
processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_PATH)

print("✅ Model loaded!")
if torch.cuda.is_available():
    print(f"💾 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")


def inference(image, question):
    """이미지와 질문을 받아 inference 수행"""
    
    if image is None:
        return "⚠️ 웹캠에서 이미지를 캡처해주세요.", "대기 중"
    
    if not question or question.strip() == "":
        question = "Describe this image in detail."
    
    try:
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            return "❌ 잘못된 이미지 형식", "에러"
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare messages
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}]
        }]
        
        # Process
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
        
        # 입력 길이 저장
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode (새로 생성된 토큰만)
        generated_ids = generated_ids[0][input_len:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = f"✅ {timestamp}"
        
        return response if response else "(빈 응답)", status
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        return f"❌ 에러: {str(e)}\n\n{error_msg}", "에러 발생"


# =========================
# Gradio UI
# =========================
with gr.Blocks(title="SmolVLM Auto Inference") as demo:
    gr.Markdown("""
    # 🎥 SmolVLM 웹캠 자동 추론 (Fine-tuned)
    
    **3초마다 자동으로 추론을 수행합니다**
    
    ### 모델 정보:
    - **Base Model**: HuggingFaceTB/SmolVLM-256M-Instruct
    - **Fine-tuned on**: Hair classification & description dataset
    - **Training**: 5 epochs, Final loss: 1.1350
    
    ### 사용 방법:
    1. 웹캠 허용 및 이미지 캡처
    2. 질문 입력
    3. "🚀 자동 추론 시작" 버튼 클릭
    4. 3초마다 자동으로 추론됩니다
    5. "⏸️ 중지" 버튼으로 멈출 수 있습니다
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 웹캠 (streaming 활성화)
            webcam = gr.Image(
                label="📷 웹캠",
                type="numpy",
                sources=["webcam"],
                streaming=True,  # 스트리밍 활성화
                height=400
            )
            
            # 질문 입력
            question = gr.Textbox(
                label="💬 질문",
                placeholder="이미지에 대해 물어보고 싶은 것을 입력하세요",
                value="Classify the hair length in this image. Possible values: short, mid, long. Output only one word.",
                lines=3
            )
            
            with gr.Row():
                start_btn = gr.Button("🚀 자동 추론 시작", variant="primary", scale=2)
                stop_btn = gr.Button("⏸️ 중지", variant="stop", scale=1)
        
        with gr.Column(scale=1):
            # 출력
            output = gr.Textbox(
                label="🤖 응답",
                lines=15,
                max_lines=20
            )
            
            # 상태
            status = gr.Textbox(
                label="📊 상태",
                value="대기 중",
                lines=1
            )
            
            # 자동 추론 상태
            auto_status = gr.Textbox(
                label="🔄 자동 추론 상태",
                value="멈춤",
                lines=1
            )
    
    # 예시 질문
    gr.Markdown("### 💡 예시 질문:")
    gr.Examples(
        examples=[
            ["Classify the hair length in this image. Possible values: short, mid, long. Output only one word."],
            ["Describe the person's hair style, color, and texture in detail."],
            ["What is the hair length? Answer in one word: short, mid, or long."],
            ["Describe what you see in this image."],
            ["이 사람의 머리 길이를 분류하세요. 가능한 값: short, mid, long"],
        ],
        inputs=[question],
    )
    
    # 자동 추론 제어
    is_auto_running = gr.State(value=False)
    last_inference_time = gr.State(value=0)
    
    def start_auto_inference():
        """자동 추론 시작"""
        # 현재 시간으로 설정하여 즉시 첫 추론 시작
        return True, "▶️ 실행 중 (3초 간격)", gr.Timer(value=0.5, active=True), time.time() - INFERENCE_INTERVAL
    
    def stop_auto_inference():
        """자동 추론 중지"""
        return False, "⏸️ 멈춤", gr.Timer(value=0.5, active=False)
    
    def auto_inference_loop(image, question_text, is_running, last_time):
        """자동 추론 루프 (3초마다 실행)"""
        if not is_running:
            return gr.update(), gr.update(), last_time
        
        current_time = time.time()
        
        # 이미지 없으면 경고 메시지
        if image is None:
            return gr.update(), "⚠️ 웹캠 이미지를 캡처해주세요", last_time
        
        # 3초 경과 확인
        if current_time - last_time >= INFERENCE_INTERVAL:
            result, status_msg = inference(image, question_text)
            return result, status_msg, current_time
        else:
            # 대기 중 남은 시간 표시
            remaining = INFERENCE_INTERVAL - (current_time - last_time)
            return gr.update(), f"⏱️ 다음 추론까지 {remaining:.1f}초", last_time
    
    # 자동 추론 타이머
    timer = gr.Timer(value=0.5, active=False)
    
    # 시작 버튼
    start_btn.click(
        fn=start_auto_inference,
        inputs=[],
        outputs=[is_auto_running, auto_status, timer, last_inference_time]
    )
    
    # 중지 버튼
    stop_btn.click(
        fn=stop_auto_inference,
        inputs=[],
        outputs=[is_auto_running, auto_status, timer]
    )
    
    # 타이머 틱
    timer.tick(
        fn=auto_inference_loop,
        inputs=[webcam, question, is_auto_running, last_inference_time],
        outputs=[output, status, last_inference_time]
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Launching at http://0.0.0.0:7860")
    print("="*70 + "\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=8085,
        share=False,
        show_error=True
    )

