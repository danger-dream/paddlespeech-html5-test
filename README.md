# paddlespeech实时音频识别测试文档

## 服务搭建：

```bash
# 使用python3.9为基础镜像
docker pull python:3.9

docker run --name=pd-base -dit python:3.9 bash

# 进容器装环境
docker exec -it pd-base bash

apt update && apt upgrade -y

# 装常用工具
apt install -y wget curl git net-tools build-essential screen jq nano locales inetutils-ping libjpeg-dev zlib1g-dev dialog

# 装paddlespeech
pip install pytest-runner
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple

# 退出容器
exit

# 导出基础镜像
docker export pd-base > pd-speech-base.tar

# 导入基础镜像
docker import pd-speech-base.tar pd-speech-base

# 在基础镜像上再进行修改
mkdir /data/pd-speech

docker run --name=pd-speech -itd -p 1000:80 -v /data/pd-speech:/pd pd-speech-base bash

docker exec -it pd-speech bash

# 从paddlespeech server中重构出实时asr语音转文本和标点符号恢复代码，优化精简后提供基于websocket的音频流预测服务
nano /pd/app.py

# docker logs -f pd-speech

# 启动服务端
python /pd/app.py

```

## 游览器打开index.html测试.........

