version="8.0.0.3"
arch=$(uname -m)
cuda="cuda-11.0"
cudnn="cudnn8.2"

tar xzvf TensorRT-${version}.Linux.${arch}-gnu.${cuda}.${cudnn}.tar.gz


ls TensorRT-${version}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>

cd TensorRT-${version}/python

sudo python3.8 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl

cd TensorRT-${version}/uff

sudo python3.8 -m pip install uff-0.6.9-py2.py3-none-any.whl

which convert-to-uff

cd TensorRT-${version}/graphsurgeon

sudo python3.8 -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

cd TensorRT-${version}/onnx_graphsurgeon

sudo python3.8 -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl