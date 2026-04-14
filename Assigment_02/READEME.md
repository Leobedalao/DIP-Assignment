# Implementation of Poisson Image Editing and Pix2Pix

This repository is Weilong Li's implementation of Assignment_02 of DIP.

## Requirements

### Task Possion Image Editing

```python
python -m pip install -r requirements.txt
```

### Task Pix2Pix

```powershell
cd ".\Pix2Pix"
uv venv
.venv\Scripts\activate
uv sync
```

## Running

### Task Possion Image Editing

```python
python run_blending_gradio.py
```

### Task Pix2Pix

运行脚本下载数据集

```bash
bash download_facades_dataset.sh
```

执行训练代码

```python
uv run python train.py
```



## Results

### Task Possion Image Editing

| ![source](./READEME.assets/source.jpg) | ![target](./READEME.assets/target.jpg) |
| -------------------------------------- | -------------------------------------- |

| ![image-20260414200910888](./READEME.assets/image-20260414200910888.png) | ![image-20260414200950837](./READEME.assets/image-20260414200950837.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

| ![source (1)](./READEME.assets/source (1).png)               | ![target (1)](./READEME.assets/target (1).png)               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20260414201304462](./READEME.assets/image-20260414201304462.png) | ![image-20260414201307943](./READEME.assets/image-20260414201307943.png) |

| <img src="./READEME.assets/source.png" alt="source" style="zoom:150%;" /> | <img src="./READEME.assets/target.png" alt="target" style="zoom:80%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

| ![image-20260414201701083](./READEME.assets/image-20260414201701083.png) | ![image-20260414201731517](./READEME.assets/image-20260414201731517.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |



### Task Pix2Pix

#### Facades

| epoch=300 | train loss=0.233176 | val loss=0.400512 |
| --------- | ------------------- | ----------------- |

| ![loss_curve](./READEME.assets/loss_curve.png) |
| ---------------------------------------------- |

训练300轮后验证集上的结果如下：

![result_1](./READEME.assets/result_1.png)

![result_2](./READEME.assets/result_2.png)

![result_3](./READEME.assets/result_3.png)

![result_4](./READEME.assets/result_4.png)

![result_5](./READEME.assets/result_5.png)

#### cityscapes

| epoch=300 | train loss=0.09794 | val loss=0.12090 |
| --------- | ------------------ | ---------------- |

| ![loss_curve](./READEME.assets/loss_curve-1776172869954-11.png) |
| ------------------------------------------------------------ |

训练300轮后验证集上的结果如下：

![result_2](./READEME.assets/result_2-1776172932271-13.png)

![result_3](./READEME.assets/result_3-1776172932271-14.png)

![result_4](./READEME.assets/result_4-1776172932271-15.png)

![result_5](./READEME.assets/result_5-1776172932271-16.png)

![result_1](./READEME.assets/result_1-1776172932271-17.png)