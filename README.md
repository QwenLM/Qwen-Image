<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_logo.png" width="400"/>
<p>
<p align="center">&nbsp&nbsp💜 <a href="https://chat.qwen.ai/">Qwen Chat</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image">HuggingFace(T2I)</a>&nbsp&nbsp |
           &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/Qwen-Image-Edit">HuggingFace(Edit)</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image">ModelScope-T2I</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/models/Qwen/Qwen-Image-Edit">ModelScope-Edit</a>&nbsp&nbsp| &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.02324">Tech Report</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image/">Blog(T2I)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen-image-edit/">Blog(Edit)</a> &nbsp&nbsp 
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image">T2I Demo</a>&nbsp&nbsp | 🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen-Image-Edit">Edit Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen-Image/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/merge3.jpg" width="1024"/>
<p>

## Introduction
We are thrilled to release **Qwen-Image**, a 20B MMDiT image foundation model that achieves significant advances in **complex text rendering** and **precise image editing**. Experiments show strong general capabilities in both image generation and editing, with exceptional performance in text rendering, especially for Chinese.


![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/bench.png#center)

## News
- 2025.08.18: We’re excited to announce the open-sourcing of Qwen-Image-Edit! 🎉 Try it out in your local environment with the quick start guide below, or head over to [Qwen Chat](https://chat.qwen.ai/) or [Huggingface Demo](https://huggingface.co/spaces/Qwen/Qwen-Image-Edit) to experience the online demo right away! If you enjoy our work, please show your support by giving our repository a star. Your encouragement means a lot to us!
- 2025.08.09: Qwen-Image now supports a variety of LoRA models, such as MajicBeauty LoRA, enabling the generation of highly realistic beauty images. Check out the available weights on [ModelScope](https://modelscope.cn/models/merjic/majicbeauty-qwen1/summary).
<p align="center">
    <img src="assets/magicbeauty.png"/>
<p>
    
- 2025.08.05: Qwen-Image is now natively supported in ComfyUI, see [Qwen-Image in ComfyUI: New Era of Text Generation in Images!](https://blog.comfy.org/p/qwen-image-in-comfyui-new-era-of)
- 2025.08.05: Qwen-Image is now on Qwen Chat. Click [Qwen Chat](https://chat.qwen.ai/) and choose "Image Generation".
- 2025.08.05: We released our [Technical Report](https://arxiv.org/abs/2508.02324) on Arxiv!
- 2025.08.04: We released Qwen-Image weights! Check at [Huggingface](https://huggingface.co/Qwen/Qwen-Image) and [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image)!
- 2025.08.04: We released Qwen-Image! Check our [Blog](https://qwenlm.github.io/blog/qwen-image) for more details!

> [!NOTE]  
> Due to heavy traffic, if you'd like to experience our demo online, we also recommend visiting DashScope, WaveSpeed, and LibLib. Please find the links below in the community support.

## Quick Start

1. Make sure your transformers>=4.51.3 (Supporting Qwen2.5-VL)

2. Install the latest version of diffusers
```
pip install git+https://github.com/huggingface/diffusers
```

### Text to Image

The following contains a code snippet illustrating how to use the model to generate images based on text prompts:

```python
from diffusers import DiffusionPipeline
import torch

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".'''

negative_prompt = " " # Recommended if you don't use a negative prompt.


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")
```

### Image Editing

```python
import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)

image = Image.open("./input.png").convert("RGB")
prompt = "Change the rabbit's color to purple, with a flash light background."


inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
```

> [!NOTE]

> We highly recommend using prompt rewriting to improve the stability of editing cases. For reference, please see our official [demo script](src/examples/edit_demo.py), which includes example system prompts. Qwen-Image-Edit is actively evolving with ongoing development. Stay tuned for future enhancements!

## Show Cases


### General Cases
One of its standout capabilities is high-fidelity text rendering across diverse images. Whether it's alphabetic languages like English or logographic scripts like Chinese, Qwen-Image preserves typographic details, layout coherence, and contextual harmony with stunning accuracy. Text isn't just overlaid, it's seamlessly integrated into the visual fabric.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s1.jpg#center)

Beyond text, Qwen-Image excels at general image generation with support for a wide range of artistic styles. From photorealistic scenes to impressionist paintings, from anime aesthetics to minimalist design, the model adapts fluidly to creative prompts, making it a versatile tool for artists, designers, and storytellers.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s2.jpg#center)

When it comes to image editing, Qwen-Image goes far beyond simple adjustments. It enables advanced operations such as style transfer, object insertion or removal, detail enhancement, text editing within images, and even human pose manipulation—all with intuitive input and coherent output. This level of control brings professional-grade editing within reach of everyday users.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s3.jpg#center)

But Qwen-Image doesn't just create or edit, it understands. It supports a suite of image understanding tasks, including object detection, semantic segmentation, depth and edge (Canny) estimation, novel view synthesis, and super-resolution. These capabilities, while technically distinct, can all be seen as specialized forms of intelligent image editing, powered by deep visual comprehension.

![](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/s4.jpg#center)

Together, these features make Qwen-Image not just a tool for generating pretty pictures, but a comprehensive foundation model for intelligent visual creation and manipulation—where language, layout, and imagery converge.

### Tutorial for Image Editing

One of the highlights of Qwen-Image-Edit lies in its powerful capabilities for semantic and appearance editing. Semantic editing refers to modifying image content while preserving the original visual semantics. To intuitively demonstrate this capability, let's take Qwen's mascot—Capybara—as an example:
![Capibara](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片3.JPG#center)
As can be seen, although most pixels in the edited image differ from those in the input image (the leftmost image), the character consistency of Capybara is perfectly preserved. Qwen-Image-Edit's powerful semantic editing capability enables effortless and diverse creation of original IP content.
Furthermore, on Qwen Chat, we designed a series of editing prompts centered around the 16 MBTI personality types. Leveraging these prompts, we successfully created a set of MBTI-themed emoji packs based on our mascot Capybara, effortlessly expanding the IP's reach and expression.
![MBTI meme series](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片4.JPG#center)
Moreover, novel view synthesis is another key application scenario in semantic editing. As shown in the two example images below, Qwen-Image-Edit can not only rotate objects by 90 degrees, but also perform a full 180-degree rotation, allowing us to directly see the back side of the object:
![Viewpoint transformation 90 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片12.JPG#center)
![Viewpoint transformation 180 degrees](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片13.JPG#center)
Another typical application of semantic editing is style transfer. For instance, given an input portrait, Qwen-Image-Edit can easily transform it into various artistic styles such as Studio Ghibli. This capability holds significant value in applications like virtual avatar creation:
![Style transfer](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片1.JPG#center)
In addition to semantic editing, appearance editing is another common image editing requirement. Appearance editing emphasizes keeping certain regions of the image completely unchanged while adding, removing, or modifying specific elements. The image below illustrates a case where a signboard is added to the scene. 
As shown, Qwen-Image-Edit not only successfully inserts the signboard but also generates a corresponding reflection, demonstrating exceptional attention to detail.
![Adding a signboard](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片6.JPG#center)
Below is another interesting example, demonstrating how to remove fine hair strands and other small objects from an image.
![Removing fine strands of hair](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片7.JPG#center)
Additionally, the color of a specific letter "n" in the image can be modified to blue, enabling precise editing of particular elements.
![Modifying text color](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片8.JPG#center)
Appearance editing also has wide-ranging applications in scenarios such as adjusting a person's background or changing clothing. The three images below demonstrate these practical use cases respectively.
![Modifying backgrounds](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片11.JPG#center)
![Modifying clothing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片5.JPG#center)
Another standout feature of Qwen-Image-Edit is its accurate text editing capability, which stems from Qwen-Image's deep expertise in text rendering. As shown below, the following two cases vividly demonstrate Qwen-Image-Edit's powerful performance in editing English text:
![Editing English text 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片15.JPG#center)
![Editing English text 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片16.JPG#center)
Qwen-Image-Edit can also directly edit Chinese posters, enabling not only modifications to large headline text but also precise adjustments to even small and intricate text elements.
![Editing Chinese posters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片17.JPG#center)
Finally, let's walk through a concrete image editing example to demonstrate how to use a chained editing approach to progressively correct errors in a calligraphy artwork generated by Qwen-Image:
![Calligraphy artwork](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片18.JPG#center)
In this artwork, several Chinese characters contain generation errors. We can leverage Qwen-Image-Edit to correct them step by step. For instance, we can draw bounding boxes on the original image to mark the regions that need correction, instructing Qwen-Image-Edit to fix these specific areas. Here, we want the character "稽" to be correctly written within the red box, and the character "亭" to be accurately rendered in the blue region.
![Correcting characters](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片19.JPG#center)
However, in practice, the character "稽" is relatively obscure, and the model fails to correct it correctly in one step. The lower-right component of "稽" should be "旨" rather than "日". At this point, we can further highlight the "日" portion with a red box, instructing Qwen-Image-Edit to fine-tune this detail and replace it with "旨".
![Fine-tuning character](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片20.JPG#center)
Isn't it amazing? With this chained, step-by-step editing approach, we can continuously correct character errors until the desired final result is achieved.
![Final version 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片21.JPG#center)
![Final version 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片22.JPG#center)
![Final version 3](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片23.JPG#center)
![Final version 4](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片24.JPG#center)
![Final version 5](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit_en/幻灯片25.JPG#center)
Finally, we have successfully obtained a completely correct calligraphy version of *Lantingji Xu (Orchid Pavilion Preface)*!
In summary, we hope that Qwen-Image-Edit can further advance the field of image generation, truly lower the technical barriers to visual content creation, and inspire even more innovative applications.




### Advanced Usage

#### Prompt Enhance
For enhanced prompt optimization and multi-language support, we recommend using our official Prompt Enhancement Tool powered by Qwen-Plus .

You can integrate it directly into your code:
```python
from tools.prompt_utils import rewrite
prompt = rewrite(prompt)
```

Alternatively, run the example script from the command line:

```bash
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx python examples/generate_w_prompt_enhance.py
```


## Deploy Qwen-Image

Qwen-Image supports Multi-GPU API Server for local deployment:

### Multi-GPU API Server Pipeline & Usage

The Multi-GPU API Server will start a Gradio-based web interface with:
- Multi-GPU parallel processing
- Queue management for high concurrency
- Automatic prompt optimization
- Support for multiple aspect ratios

Configuration via environment variables:
```bash
export NUM_GPUS_TO_USE=4          # Number of GPUs to use
export TASK_QUEUE_SIZE=100        # Task queue size
export TASK_TIMEOUT=300           # Task timeout in seconds
```

```bash
# Start the gradio demo server, api key for prompt enhance
cd src
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxx python examples/demo.py 
```


## AI Arena

To comprehensively evaluate the general image generation capabilities of Qwen-Image and objectively compare it with state-of-the-art closed-source APIs, we introduce [AI Arena](https://aiarena.alibaba-inc.com), an open benchmarking platform built on the Elo rating system. AI Arena provides a fair, transparent, and dynamic environment for model evaluation.

In each round, two images—generated by randomly selected models from the same prompt—are anonymously presented to users for pairwise comparison. Users vote for the better image, and the results are used to update both personal and global leaderboards via the Elo algorithm, enabling developers, researchers, and the public to assess model performance in a robust and data-driven way. AI Arena is now publicly available, welcoming everyone to participate in model evaluations. 

![AI Arena](assets/figure_aiarena_website.png)

The latest leaderboard rankings can be viewed at [AI Arena Learboard](https://aiarena.alibaba-inc.com/corpora/arena/leaderboard?arenaType=text2image).

If you wish to deploy your model on AI Arena and participate in the evaluation, please contact weiyue.wy@alibaba-inc.com.

## Community Support

### Huggingface

Diffusers has supported Qwen-Image since day 0. Support for LoRA and finetuning workflows is currently in development and will be available soon.

### ModelScope
* **[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)** provides comprehensive support for Qwen-Image, including low-GPU-memory layer-by-layer offload (inference within 4GB VRAM), FP8 quantization, LoRA / full training.
* **[DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)** delivers advanced optimizations for Qwen-Image inference and deployment, including FBCache-based acceleration, classifier-free guidance (CFG) parallel, and more.
* **[ModelScope AIGC Central](https://www.modelscope.cn/aigc)** provides hands-on experiences on Qwen Image, including: 
    - [Image Generation](https://www.modelscope.cn/aigc/imageGeneration): Generate high fidelity images using the Qwen Image model.
    - [LoRA Training](https://www.modelscope.cn/aigc/modelTraining): Easily train Qwen Image LoRAs for personalized concepts.

### WaveSpeedAI

WaveSpeed has deployed Qwen-Image on their platform from day 0, visit their [model page](https://wavespeed.ai/models/wavespeed-ai/qwen-image/text-to-image) for more details.

### LiblibAI

LiblibAI offers native support for Qwen-Image from day 0. Visit their [community](https://www.liblib.art/modelinfo/c62a103bd98a4246a2334e2d952f7b21?from=sd&versionUuid=75e0be0c93b34dd8baeec9c968013e0c) page for more details and discussions.

### Inference Acceleration Method: cache-dit

cache-dit offers cache acceleration support for Qwen-Image with DBCache, TaylorSeer and Cache CFG. Visit their [example](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py) for more details.

## License Agreement

Qwen-Image is licensed under Apache 2.0. 

## Citation

We kindly encourage citation of our work if you find it useful.

```bibtex
@misc{wu2025qwenimagetechnicalreport,
      title={Qwen-Image Technical Report}, 
      author={Chenfei Wu and Jiahao Li and Jingren Zhou and Junyang Lin and Kaiyuan Gao and Kun Yan and Sheng-ming Yin and Shuai Bai and Xiao Xu and Yilei Chen and Yuxiang Chen and Zecheng Tang and Zekai Zhang and Zhengyi Wang and An Yang and Bowen Yu and Chen Cheng and Dayiheng Liu and Deqing Li and Hang Zhang and Hao Meng and Hu Wei and Jingyuan Ni and Kai Chen and Kuan Cao and Liang Peng and Lin Qu and Minggang Wu and Peng Wang and Shuting Yu and Tingkun Wen and Wensen Feng and Xiaoxiao Xu and Yi Wang and Yichang Zhang and Yongqiang Zhu and Yujia Wu and Yuxuan Cai and Zenan Liu},
      year={2025},
      eprint={2508.02324},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.02324}, 
}
```


## Contact and Join Us


If you'd like to get in touch with our research team, we'd love to hear from you! Join our [Discord](https://discord.gg/z3GAxXZ9Ce) or scan the QR code to connect via our [WeChat groups](assets/wechat.png) — we're always open to discussion and collaboration.

If you have questions about this repository, feedback to share, or want to contribute directly, we welcome your issues and pull requests on GitHub. Your contributions help make Qwen-Image better for everyone. 

If you're passionate about fundamental research, we're hiring full-time employees (FTEs) and research interns. Don't wait — reach out to us at fulai.hr@alibaba-inc.com

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-Image&type=Date)](https://www.star-history.com/#QwenLM/Qwen-Image&Date)










