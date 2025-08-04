from huggingface_hub import snapshot_download, login
import os

model_name="Qwen/Qwen-Image"
folder_name = model_name.split('/')[-1]

def download_model(model_name):
    snapshot_download(
        repo_id=model_name,
        local_dir=f"models/{folder_name}",
        allow_patterns=[
            "*",
        ],
        resume_download=True
    )
    print("下载成功！")

try:
    download_model(model_name)
    
except Exception as e:
    if "401" in str(e):
        # 2. 需要登录
        print("需要 Hugging Face 登录，创建新令牌（选择'Read'权限即可）")
        token = input("请输入您的 Hugging Face 访问令牌 (https://huggingface.co/settings/tokens): ")
        
        # 登录并重试
        login(token=token)
        download_model(model_name)
        
    else:
        print(f"下载失败: {str(e)}")