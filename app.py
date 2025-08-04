import torch
import argparse
import os
import numpy as np
import datetime
import random
from diffusers import DiffusionPipeline
import gradio as gr
from optimum.quanto import freeze, qint8, quantize

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument('--vram', type=str, default='high', choices=['low', 'high'], help='显存模式')
parser.add_argument('--lora', type=str, default="None", help='lora模型路径')
args = parser.parse_args()


if torch.cuda.is_available():
    device = "cuda" 
    if torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32


MAX_SEED = np.iinfo(np.int32).max
os.makedirs("outputs", exist_ok=True)
model_id = "models/Qwen-Image"
pipe = DiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=dtype,
)
if args.lora!="None":
    pipe.load_lora_weights(args.lora)
    print(f"加载{args.lora}")
if args.vram=="high":
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
else:
    quantize(pipe.transformer, qint8)
    freeze(pipe.transformer)
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()


def generate(
    prompt,
    negative_prompt,
    width,
    height,
    num_inference_steps,
    true_cfg_scale, 
    seed_param,
):
    global pipe, model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if seed_param<0:
        seed = random.randint(0, MAX_SEED)
    else:
        seed = seed_param
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        generator=torch.Generator().manual_seed(seed)
    ).images[0]
    image.save(f"outputs/{timestamp}.png")
    return f"outputs/{timestamp}.png", seed
    

with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Qwen-Image</h2>
            </div>
            """)
    with gr.TabItem("Qwen-Image文生图"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt = gr.Textbox(label="负面提示词", value="")
                width = gr.Slider(label="宽度（推荐1328x1328、1664x928、1472x1140）", minimum=256, maximum=2656, step=32, value=1328)
                height = gr.Slider(label="高度", minimum=256, maximum=2656, step=32, value=1328)
                num_inference_steps = gr.Slider(label="采样步数", minimum=1, maximum=100, step=1, value=50)
                true_cfg_scale = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=4.0)
                seed_param = gr.Number(label="种子，请输入正整数，-1为随机", value=-1)
                generate_button = gr.Button("🎬 开始生成", variant='primary')
            with gr.Column():
                image_output = gr.Image(label="生成图片")
                seed_output = gr.Textbox(label="种子")

    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate,
        inputs = [
            prompt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            true_cfg_scale, 
            seed_param,
        ],
        outputs = [image_output, seed_output]
    )

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )