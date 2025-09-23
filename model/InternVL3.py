# model/InternVL3.py
import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText


class InternVL3_1B:
    """
    Minimal adapter to run InternVL3-1B with LVLM-LP's run_model.py.
    Exposes:
      - forward(image_np, prompt) -> str
      - forward_with_probs(image_np, prompt) -> (str, output_ids, logits, probs)
    Where:
      * image_np: np.ndarray (RGB) as loaded by cv2 then converted to RGB in run_model.py
      * output_ids: 1D tensor of generated token ids (only new tokens, excludes prompt)
      * logits/probs: shape [gen_len, vocab_size], first row是首token的logits/probs
    """

    def __init__(self, args):
        self.args = args
        model_id = getattr(args, "model_path", "OpenGVLab/InternVL3-1B-hf")
        # bfloat16 更稳：A100/4090/MI300 等都支持；CPU 回落到 float32
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # InternVL3-1B 的 HF 原生实现：AutoProcessor + AutoModelForImageTextToText
        # 参考官方用法：apply_chat_template -> generate -> processor.decode
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map="auto",
        ).eval()

    def _prepare_inputs(self, image_np, prompt: str):
        """把 numpy(RGB)/PIL.Image 转为 HF 的 chat 模板输入，并放到正确设备/精度。"""
        if isinstance(image_np, np.ndarray):
            pil = Image.fromarray(image_np)  # run_model.py 已经 BGR->RGB
        elif isinstance(image_np, Image.Image):
            pil = image_np
        else:
            raise TypeError("image must be a numpy.ndarray (RGB) or PIL.Image.Image")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # 直接由 processor 生成可喂给 generate 的字典（含 input_ids / pixel_values 等）
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # 移动到模型设备与 dtype
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self.model.device, dtype=self.dtype)
        return inputs

    @torch.inference_mode()
    def forward(self, image_np, prompt: str):
        """纯生成：只返回文本（给 GPT4V 分支以外的统一接口）。"""
        inputs = self._prepare_inputs(image_np, prompt)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=64,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            num_beams=self.args.num_beams,
            do_sample=(self.args.temperature is not None and self.args.temperature > 0),
        )
        gen_ids = self.model.generate(**inputs, **gen_kwargs)
        text = self.processor.decode(gen_ids[0, input_len:], skip_special_tokens=True)
        return text

    @torch.inference_mode()
    def forward_with_probs(self, image_np, prompt: str):
        """
        生成并返回：
          response(str),
          output_ids(首个新token开始的 id 序列，torch.LongTensor, shape [gen_len]),
          logits(torch.FloatTensor, shape [gen_len, vocab]),
          probs(torch.FloatTensor,  shape [gen_len, vocab]).
        注意：run_model.py 只会把第 args.token_id 个时间步（通常是 0）写入 jsonl。
        """
        inputs = self._prepare_inputs(image_np, prompt)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=64,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            num_beams=self.args.num_beams,
            do_sample=(self.args.temperature is not None and self.args.temperature > 0),
            return_dict_in_generate=True,
            output_scores=True,   # 关键：拿到每步 logits
        )
        out = self.model.generate(**inputs, **gen_kwargs)

        # 只取“新生成”的 token ids（去掉提示部分）
        sequences = out.sequences            # [1, input_len + gen_len]
        gen_token_ids = sequences[0, input_len:]  # [gen_len]

        # out.scores 是 list[len=gen_len]，每个元素是 [batch, vocab] 的logits
        if out.scores and len(out.scores) > 0:
            step_logits = torch.stack([s[0].to(torch.float32) for s in out.scores], dim=0)  # [gen_len, vocab]
            step_probs = torch.softmax(step_logits, dim=-1)
        else:
            vocab = self.model.get_output_embeddings().weight.shape[0]
            step_logits = torch.zeros((0, vocab), dtype=torch.float32, device=self.model.device)
            step_probs = torch.zeros_like(step_logits)

        text = self.processor.decode(gen_token_ids, skip_special_tokens=True)

        # run_model.py 会 .tolist()，这里返回 CPU tensor 即可
        return (
            text,
            gen_token_ids.detach().cpu(),
            step_logits.detach().cpu(),
            step_probs.detach().cpu(),
        )
