import gradio as gr
import numpy as np
import random
import os
import json
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.multiprocessing as mp
from multiprocessing import Process, Queue, Event
import atexit
import signal

mp.set_start_method('spawn', force=True)

from diffusers import DiffusionPipeline
import torch
from tools.prompt_utils import rewrite

model_repo_id = "Qwen/Qwen-Image"
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1440

NUM_GPUS_TO_USE = int(os.environ.get("NUM_GPUS_TO_USE", torch.cuda.device_count()))  
TASK_QUEUE_SIZE = int(os.environ.get("TASK_QUEUE_SIZE", 100))  
TASK_TIMEOUT = int(os.environ.get("TASK_TIMEOUT", 300))  

print(f"Config: Using {NUM_GPUS_TO_USE} GPUs, queue size {TASK_QUEUE_SIZE}, timeout {TASK_TIMEOUT} seconds")


class GPUWorker:
    def __init__(self, gpu_id, model_repo_id, task_queue, result_queue, stop_event):
        self.gpu_id = gpu_id
        self.model_repo_id = model_repo_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.device = f"cuda:{gpu_id}"
        self.pipe = None
        
    def initialize_model(self):
        """Initialize the model on the specified GPU"""
        try:
            torch.cuda.set_device(self.gpu_id)
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
            
            self.pipe = DiffusionPipeline.from_pretrained(self.model_repo_id, torch_dtype=torch_dtype)
            self.pipe = self.pipe.to(self.device)
            print(f"GPU {self.gpu_id} model initialized successfully")
            return True
        except Exception as e:
            print(f"GPU {self.gpu_id} model initialization failed: {e}")
            return False
    
    def process_task(self, task):
        """Process a single task"""
        try:
            task_id = task['task_id']
            prompt = task['prompt']
            negative_prompt = task['negative_prompt']
            seed = task['seed']
            width = task['width']
            height = task['height']
            guidance_scale = task['guidance_scale']
            num_inference_steps = task['num_inference_steps']
            progress_callback = task['progress_callback']
            
            def step_callback(pipe, i, t, callback_kwargs):
                progress_callback(0.2 + i / num_inference_steps * 0.8, desc="GPU processing...")
                return callback_kwargs
            
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            with torch.cuda.device(self.gpu_id):
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    true_cfg_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=step_callback
                ).images[0]
            
            return {
                'task_id': task_id,
                'image': image,
                'success': True,
                'gpu_id': self.gpu_id
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'gpu_id': self.gpu_id
            }
    
    def run(self):
        """Worker main loop"""
        if not self.initialize_model():
            return
        
        print(f"GPU {self.gpu_id} worker starting")
        
        while not self.stop_event.is_set():
            try:
                # Get task from the task queue, set timeout to check stop event
                task = self.task_queue.get(timeout=1)
                if task is None:  # Poison pill, exit signal
                    break
                
                # Process the task
                result = self.process_task(task)
                
                # Put the result into the result queue
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU {self.gpu_id} worker exception: {e}")
                continue
        
        print(f"GPU {self.gpu_id} worker stopping")

# Global GPU worker function for spawn mode
def gpu_worker_process(gpu_id, model_repo_id, task_queue, result_queue, stop_event):
    worker = GPUWorker(gpu_id, model_repo_id, task_queue, result_queue, stop_event)
    worker.run()

# Multi-GPU Manager Class
class MultiGPUManager:
    def __init__(self, model_repo_id, num_gpus=None, task_queue_size=100):
        self.model_repo_id = model_repo_id
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.task_queue = Queue(maxsize=task_queue_size)  
        self.result_queue = Queue()  
        self.stop_event = Event()
        self.workers = []
        self.worker_processes = []
        self.task_counter = 0
        self.pending_tasks = {}  
        
        print(f"Initializing Multi-GPU Manager with {self.num_gpus} GPUs, queue size {task_queue_size}")
        
    def start_workers(self):
        """Start all GPU workers"""
        for gpu_id in range(self.num_gpus):
            # Use global function instead of instance method to ensure proper operation in spawn mode
            process = Process(target=gpu_worker_process, 
                            args=(gpu_id, self.model_repo_id, self.task_queue, 
                                  self.result_queue, self.stop_event))
            process.start()
            
            self.worker_processes.append(process)
        
        # Start result processing thread
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        print(f"All {self.num_gpus} GPU workers have started")
    
    def _process_results(self):
        """Background thread for processing results"""
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=1)
                task_id = result['task_id']
                
                if task_id in self.pending_tasks:
                    # Pass the result to the waiting task
                    self.pending_tasks[task_id]['result'] = result
                    self.pending_tasks[task_id]['event'].set()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Result processing thread exception: {e}")
                continue
    
    def submit_task(self, prompt, negative_prompt="", seed=42, width=1664, height=928, 
                   guidance_scale=4, num_inference_steps=50, timeout=300):
        """Submit task and wait for result"""
        return self.submit_task_with_progress(prompt, negative_prompt, seed, width, height, 
                                            guidance_scale, num_inference_steps, timeout, None)
    
    def submit_task_with_progress(self, prompt, negative_prompt="", seed=42, width=1664, height=928, 
                                 guidance_scale=4, num_inference_steps=50, timeout=300, progress_callback=None):
        """Submit task and wait for result, with progress callback support"""
        task_id = f"task_{self.task_counter}_{time.time()}"
        self.task_counter += 1
        
        task = {
            'task_id': task_id,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'width': width,
            'height': height,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'progress_callback': progress_callback
        }
        
        # Create waiting event
        result_event = threading.Event()
        self.pending_tasks[task_id] = {
            'event': result_event,
            'result': None,
            'submitted_time': time.time()
        }
        
        try:
            # Put task into queue
            self.task_queue.put(task, timeout=10)
            
            if progress_callback:
                progress_callback(0.2, desc="Task submitted, waiting for GPU processing...")
            
            # Wait for result, with progress update
            start_time = time.time()
            while not result_event.is_set():
                if progress_callback:
                    elapsed = time.time() - start_time
                    # Estimate progress (between 40% and 80%)
                    estimated_progress = 0.2 + min(0.4, (elapsed / (timeout * 0.5)) * 0.4)
                    # progress_callback(estimated_progress, desc="GPU processing...")
                
                if result_event.wait(timeout=2):  # Check every 2 seconds
                    break
                    
                if time.time() - start_time > timeout:
                    # Timeout
                    del self.pending_tasks[task_id]
                    return {'success': False, 'error': 'Task timeout'}
            
            if progress_callback:
                progress_callback(0.8, desc="GPU processing complete...")
            
            result = self.pending_tasks[task_id]['result']
            del self.pending_tasks[task_id]
            return result
                
        except queue.Full:
            del self.pending_tasks[task_id]
            return {'success': False, 'error': 'Task queue is full'}
        except Exception as e:
            if task_id in self.pending_tasks:
                del self.pending_tasks[task_id]
            return {'success': False, 'error': str(e)}
    
    def get_queue_status(self):
        """Get queue status"""
        return {
            'task_queue_size': self.task_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'pending_tasks': len(self.pending_tasks),
            'active_workers': len(self.worker_processes)
        }
    
    def stop(self):
        """Stop all workers"""
        print("Stopping Multi-GPU Manager...")
        self.stop_event.set()
        
        # Send stop signal to each worker
        for _ in range(self.num_gpus):
            try:
                self.task_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Wait for all processes to end
        for process in self.worker_processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        print("Multi-GPU Manager has stopped")

# Global Multi-GPU Manager instance
gpu_manager = None

def initialize_gpu_manager():
    """Initialize global GPU manager"""
    global gpu_manager
    if gpu_manager is None:
        try:
            # Ensure main process does not initialize CUDA
            if torch.cuda.is_available():
                print(f"Detected {torch.cuda.device_count()} GPUs")
            
            gpu_manager = MultiGPUManager(
                model_repo_id, 
                num_gpus=NUM_GPUS_TO_USE,
                task_queue_size=TASK_QUEUE_SIZE
            )
            gpu_manager.start_workers()
            print("GPU Manager initialized successfully")
        except Exception as e:
            print(f"GPU Manager initialization failed: {e}")
            gpu_manager = None

# Lazy initialization, only initialize when needed
gpu_manager = None


# (1664, 928), (1472, 1104), (1328, 1328)
def get_image_size(aspect_ratio):
    if aspect_ratio == "1:1":
        return 1328, 1328
    elif aspect_ratio == "16:9":
        return 1664, 928
    elif aspect_ratio == "9:16":
        return 928, 1664
    elif aspect_ratio == "4:3":
        return 1472, 1104
    elif aspect_ratio == "3:4":
        return 1104, 1472
    else:
        return 1328, 1328


def infer(
    prompt,
    negative_prompt="",
    seed=42,
    randomize_seed=False,
    aspect_ratio="16:9",
    guidance_scale=5,
    num_inference_steps=50,
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
):
    global gpu_manager
    
    # Lazy load GPU manager
    if gpu_manager is None:
        progress(0.1, desc="Initializing GPU manager...")
        initialize_gpu_manager()
        
        # Return error if initialization fails
        if gpu_manager is None:
            print("GPU manager initialization failed, unable to process task")
            from PIL import Image
            error_image = Image.new('RGB', (512, 512), color='gray')
            return error_image, seed

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    width, height = get_image_size(aspect_ratio)
    original_prompt = prompt
    
    # Rewrite prompt
    progress(0.1, desc="Optimizing prompt...")
    prompt = rewrite(prompt)
    print(f"Prompt: {prompt}, original_prompt: {original_prompt}")

    # Submit task to queue
    progress(0.3, desc="Submitting task to GPU queue...")
    
    # Submit task using global GPU manager with progress tracking
    result = gpu_manager.submit_task_with_progress(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        timeout=TASK_TIMEOUT,
        progress_callback=progress,
    )

    if result['success']:
        progress(0.9, desc="Saving result...")
        image = result['image']
        gpu_id = result['gpu_id']
        print(f"Task completed using GPU {gpu_id}")

        progress(1.0, desc="Done!")
        return image, seed
    else:
        print(f"Inference failed: {result['error']}")
        # Return a blank image or error message
        from PIL import Image
        error_image = Image.new('RGB', (512, 512), color='red')
        return error_image, seed


def get_system_status():
    """Get system status"""
    if gpu_manager:
        status = gpu_manager.get_queue_status()
        return f"""
        ## System Status
        - Active Workers: {status['active_workers']}
        - Task Queue Size: {status['task_queue_size']}
        - Result Queue Size: {status['result_queue_size']}
        - Pending Tasks: {status['pending_tasks']}
        - Total GPUs: {gpu_manager.num_gpus}
        """
    else:
        return "GPU manager not initialized"

examples = [
        "A capybara wearing a suit holding a sign that reads Hello World",
        "一幅精致细腻的工笔画，画面中心是一株蓬勃生长的红色牡丹，花朵繁茂，既有盛开的硕大花瓣，也有含苞待放的花蕾，层次丰富，色彩艳丽而不失典雅。牡丹枝叶舒展，叶片浓绿饱满，脉络清晰可见，与红花相映成趣。一只蓝紫色蝴蝶仿佛被画中花朵吸引，停驻在画面中央的一朵盛开牡丹上，流连忘返，蝶翼轻展，细节逼真，仿佛随时会随风飞舞。整幅画作笔触工整严谨，色彩浓郁鲜明，展现出中国传统工笔画的精妙与神韵，画面充满生机与灵动之感。",
        "一位身着淡雅水粉色交领襦裙的年轻女子背对镜头而坐，俯身专注地手持毛笔在素白宣纸上书写“通義千問”四个遒劲汉字。古色古香的室内陈设典雅考究，案头错落摆放着青瓷茶盏与鎏金香炉，一缕熏香轻盈升腾；柔和光线洒落肩头，勾勒出她衣裙的柔美质感与专注神情，仿佛凝固了一段宁静温润的旧时光。",
        " 一个可抽取式的纸巾盒子，上面写着'Face, CLEAN & SOFT TISSUE'下面写着'亲肤可湿水'，左上角是品牌名'洁柔'，整体是白色和浅黄色的色调",
        "手绘风格的水循环示意图，整体画面呈现出一幅生动形象的水循环过程图解。画面中央是一片起伏的山脉和山谷，山谷中流淌着一条清澈的河流，河流最终汇入一片广阔的海洋。山体和陆地上绘制有绿色植被。画面下方为地下水层，用蓝色渐变色块表现，与地表水形成层次分明的空间关系。太阳位于画面右上角，促使地表水蒸发，用上升的曲线箭头表示蒸发过程。云朵漂浮在空中，由白色棉絮状绘制而成，部分云层厚重，表示水汽凝结成雨，用向下箭头连接表示降雨过程。雨水以蓝色线条和点状符号表示，从云中落下，补充河流与地下水。整幅图以卡通手绘风格呈现，线条柔和，色彩明亮，标注清晰。背景为浅黄色纸张质感，带有轻微的手绘纹理。",
        '一个会议室，墙上写着"3.14159265-358979-32384626-4338327950"，一个小陀螺在桌上转动',
        '一个咖啡点门口有一个黑板，上面写着通义千问咖啡，2美元一杯，旁边有个霓虹灯，写着阿里巴巴，旁边有个海报，海报上面是一个中国美女，海报下方写着qwen newbee',
        """A young girl wearing school uniform stands in a classroom, writing on a chalkboard. The text "Introducing Qwen-Image, a foundational image generation model that excels in complex text rendering and precise image editing" appears in neat white chalk at the center of the blackboard. Soft natural light filters through windows, casting gentle shadows. The scene is rendered in a realistic photography style with fine details, shallow depth of field, and warm tones. The girl's focused expression and chalk dust in the air add dynamism. Background elements include desks and educational posters, subtly blurred to emphasize the central action. Ultra-detailed 32K resolution, DSLR-quality, soft bokeh effect, documentary-style composition""",
        "Realistic still life photography style: A single, fresh apple resting on a clean, soft-textured surface. The apple is slightly off-center, softly backlit to highlight its natural gloss and subtle color gradients—deep crimson red blending into light golden hues. Fine details such as small blemishes, dew drops, and a few light highlights enhance its lifelike appearance. A shallow depth of field gently blurs the neutral background, drawing full attention to the apple. Hyper-detailed 8K resolution, studio lighting, photorealistic render, emphasizing texture and form."
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown('[![](./avatar.png)](https://huggingface.co/Qwen/Qwen-Image)')
        gr.Markdown(" # [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)")
        gr.Markdown("[Learn more](https://huggingface.co/Qwen/Qwen-Image) about the Qwen-Image series. Try on [Hugging Face API](https://huggingface.co/Qwen/Qwen-Image), or [download model](https://huggingface.co/Qwen/Qwen-Image) to run locally with ComfyUI or diffusers.")
        gr.Markdown("**For better results when generating images with text, try enclosing the text you want in quotation marks like this: \"text you want\"**")
        gr.Markdown("**如果想在生成图像时获得更好的文字效果，建议将你想要的文字用引号括起来，例如：\"你想要的文字\"。**")
        
        with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0, variant="primary")
        
        result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=False):
            negative_prompt = gr.Text(
                label="Negative prompt",
                max_lines=1,
                placeholder="Enter a negative prompt",
                visible=False,
            )

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():
                aspect_ratio = gr.Radio(
                    label="Aspect ratio(width:height)",
                    choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
                    value="16:9",
                )

            with gr.Row():
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.0,
                    maximum=7.5,
                    step=0.1,
                    value=4.0,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=50, 
                )

        gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False, cache_mode="lazy")
    
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            aspect_ratio,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=[result, seed],
        concurrency_limit=NUM_GPUS_TO_USE
    )
    

if __name__ == "__main__":
    def cleanup():
        if gpu_manager:
            gpu_manager.stop()
    
    # Register cleanup function
    atexit.register(cleanup)
    
    # Handle signals
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, cleaning up resources...")
        cleanup()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        demo.launch(server_name="0.0.0.0")
    except KeyboardInterrupt:
        print("Received interrupt signal, cleaning up resources...")
        cleanup()
    except Exception as e:
        print(f"Application exception: {e}")
        cleanup()
        raise