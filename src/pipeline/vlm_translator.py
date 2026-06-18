import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Định nghĩa cấu trúc dữ liệu JSON đầu ra
# ==========================================
class SortedBox(BaseModel):
    box_id: int = Field(description="ID của hộp chữ gốc (index)")
    sentence_id: int = Field(
        description="ID của câu/đoạn văn chứa hộp chữ này. Các hộp thuộc cùng một câu liên tục phải có cùng sentence_id."
    )

class LayoutReconstruction(BaseModel):
    reading_order: List[int] = Field(
        description="Mảng chứa các box_id được sắp xếp theo thứ tự đọc tự nhiên từ trên xuống dưới, trái sang phải."
    )
    boxes: List[SortedBox] = Field(
        description="Danh sách thông tin nhóm câu của các hộp chữ."
    )

# ==========================================
# 2. Lớp bổ trợ phân tích bố cục bằng LLM (Text-only)
# ==========================================
class GeminiLayoutHelper:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        self.llm = None
        self.structured_llm = None
        
        if self.api_key:
            try:
                # Khởi tạo model Gemini qua LangChain ở chế độ Text-only
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                self.structured_llm = self.llm.with_structured_output(LayoutReconstruction)
            except Exception as exc:
                print(f"[GeminiLayoutHelper] Không thể khởi tạo LangChain: {exc}")
                self.llm = None

    @property
    def is_available(self) -> bool:
        """Kiểm tra xem API có sẵn sàng chạy hay không"""
        return self.structured_llm is not None

    def reconstruct_layout(self, regions: List[Dict[str, Any]]) -> Optional[LayoutReconstruction]:
        """
        Gửi thông tin text và tọa độ hộp chữ của OCR sang Gemini để phân tích thứ tự đọc.
        Hoàn toàn không gửi hình ảnh (Text-only) giúp tiết kiệm 98% chi phí API.
        """
        if not self.is_available or not regions:
            return None

        # Trích xuất dữ liệu gọn nhẹ để gửi
        box_data = []
        for r in regions:
            box_data.append({
                "id": r["index"],
                "text": r.get("detector_text", "").strip(),
                "box": [round(c, 1) for c in r["box"]]  # [x1, y1, x2, y2]
            })

        prompt = """
        Bạn là một chuyên gia phân tích bố cục văn bản. Dưới đây là danh sách các hộp chữ được trích xuất từ một bức ảnh bằng OCR, bao gồm ID (index), nội dung văn bản tiếng Anh và tọa độ bounding box [x1, y1, x2, y2].

        Hãy thực hiện:
        1. Phân tích bố cục thực tế và sắp xếp lại danh sách ID này theo đúng thứ tự đọc tự nhiên nhất của con người (từ trên xuống dưới, từ trái sang phải, ưu tiên đọc hết từng cột nếu là dạng bố cục nhiều cột).
        2. Nhóm các hộp chữ thuộc cùng một câu hoặc cùng một đoạn văn liền mạch với nhau bằng cách gán chung một `sentence_id` (bắt đầu từ 0). 
           - Ví dụ: nếu một câu dài được ngắt thành 3 dòng, cả 3 hộp chứa dòng đó phải có chung `sentence_id`.

        Dữ liệu đầu vào:
        {box_data_json}
        """

        try:
            formatted_prompt = prompt.format(box_data_json=json.dumps(box_data, ensure_ascii=False, indent=2))
            # Gọi LLM xử lý
            result: LayoutReconstruction = self.structured_llm.invoke(formatted_prompt)
            return result
        except Exception as exc:
            print(f"[GeminiLayoutHelper] API Call failed: {exc}. Chuyển sang fallback thuật toán local.")
            return None
