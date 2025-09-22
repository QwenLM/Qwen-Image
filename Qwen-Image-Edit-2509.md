# Qwen-Image-Edit-2509 Introduction

This September, we are pleased to introduce Qwen-Image-Edit-2509, the monthly iteration of Qwen-Image-Edit. To experience the latest model, please visit [Qwen Chat](https://qwen.ai)  and select the "Image Editing" feature.

Compared with Qwen-Image-Edit released in August, the main improvements of Qwen-Image-Edit-2509 include:

* **Multi-image Editing Support**: For multi-image inputs, Qwen-Image-Edit-2509 builds upon the Qwen-Image-Edit architecture and is further trained via image concatenation to enable multi-image editing. It supports various combinations such as "person + person," "person + product," and "person + scene." Optimal performance is currently achieved with 1 to 3 input images.

* **Enhanced Single-image Consistency**: For single-image inputs, Qwen-Image-Edit-2509 significantly improves consistency, specifically in the following areas:
  - **Improved Person Editing Consistency**: Better preservation of facial identity, supporting various portrait styles and pose transformations;
  - **Improved Product Editing Consistency**: Better preservation of product identity, supporting product poster editing；
  - **Improved Text Editing Consistency**: In addition to modifying text content, it also supports editing text fonts, colors, and materials；

* **Native Support for ControlNet**: Including depth maps, edge maps, keypoint maps, and more.


## Example Showcase

**The primary update in Qwen-Image-Edit-2509 is support for multi-image inputs.**

Let’s first look at a "person + person" example:  
![Person + Person](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片19.JPG#center)

Here is a "person + scene" example:  
![Person + Scene](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片20.JPG#center)

Below is a "person + object" example:  
![Person + Object](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片21.JPG#center)

In fact, multi-image input also supports commonly used ControlNet keypoint maps—for example, changing a person’s pose:  
![Keypoint Pose Change](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片22.JPG#center)

Similarly, the following examples demonstrate results using three input images:  
![Three Images 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片23.JPG#center)  
![Three Images 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片24.JPG#center)  
![Three Images 3](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片25.JPG#center)

---

**Another major update in Qwen-Image-Edit-2509 is enhanced consistency.**

First, regarding person consistency, Qwen-Image-Edit-2509 shows significant improvement over Qwen-Image-Edit. Below are examples generating various portrait styles:  
![Portrait Styles](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片1.JPG#center)

For instance, changing a person’s pose while maintaining excellent identity consistency:  
![Pose Change with Identity](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片2.JPG#center)

Leveraging this improvement along with Qwen-Image’s unique text rendering capability, we find that Qwen-Image-Edit-2509 excels at creating meme images:  
![Meme Generation](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片3.JPG#center)

Of course, even with longer text, Qwen-Image-Edit-2509 can still render it while preserving the person’s identity:  
![Long Text with Identity](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片4.JPG#center)

Person consistency is also evident in old photo restoration. Below are two examples:  
![Old Photo Restoration 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片17.JPG#center)  
![Old Photo Restoration 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片18.JPG#center)

Naturally, besides real people, generating cartoon characters and cultural creations is also possible:  
![Cartoon & Cultural Creation](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片15.JPG#center)

Second, Qwen-Image-Edit-2509 specifically enhances product consistency. We find that the model can naturally generate product posters from plain-background product images:  
![Product Poster](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片5.JPG#center)

Or even simple logos:  
![Logo Generation](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片16.JPG#center)

Third, Qwen-Image-Edit-2509 specifically enhances text consistency and supports editing font type, font color, and font material:  
![Font Type](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片10.JPG#center)  
![Font Color](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片11.JPG#center)  
![Font Material](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片12.JPG#center)

Moreover, the ability for precise text editing has been significantly enhanced:  
![Precise Text Editing 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片13.JPG#center)  
![Precise Text Editing 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片14.JPG#center)

It is worth noting that text editing can often be seamlessly integrated with image editing—for example, in this poster editing case:  
![Poster Editing](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片6.JPG#center)

---

**The final update in Qwen-Image-Edit-2509 is native support for commonly used ControlNet image conditions, such as keypoint control and sketches:**  
![Keypoint Control](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片7.JPG#center)  
![Sketch Input 1](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片8.JPG#center)  
![Sketch Input 2](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/幻灯片9.JPG#center)

---

The above summarizes the main enhancements in this update. We hope you enjoy using Qwen-Image-Edit-2509!