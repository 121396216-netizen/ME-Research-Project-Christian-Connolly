#!/bin/bash
export LD_LIBRARY_PATH=/root/christian/lm_stuff/llama.cpp/build_opencl/bin
/root/christian/lm_stuff/llama.cpp/build_opencl/bin/llama-server --model /root/christian/lm_stuff/models/LFM2-1.2B-Tool-Q4_0.gguf --host 127.0.0.1 --port 8081 --n-gpu-layers 999 --ctx-size 2048
