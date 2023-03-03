# coding=UTF-8
from torch.autograd import Variable
import torch.onnx
import torchvision
import os
from onnx import optimizer
import sys
import onnx
import argparse
#from onnx_tf.backend import prepare
os.environ["CUDA_VISIBLE_DEVICES"]="0"



def pytorch2onnx(torch_path, onnx_path):
    model = torch.load(torch_path)
    # print(model)

    # model.load_state_dict(net).cuda()
    dummy_input = Variable(torch.randn((1, 2, 64, 64), device='cuda'))
    # Export the model
    torch.onnx.export(model,               # model being run
                      dummy_input,                         # model input (or a tuple for multiple inputs)
                      onnx_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=True,
                      keep_initializers_as_inputs=True,
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},
                        'output' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}
                        })

def printnode(onnx_path):
    print("onnx_model_path", onnx_path)
    original_model = onnx.load(onnx_path)
    print("checker")
    onnx.checker.check_model(original_model)
    print("model inputs/outputs:")
    print(original_model.graph.input[0])
    print(original_model.graph.output[0])


def optimize(onnx_path, onnx_out_path):
    original_model = onnx.load(onnx_path)

    # A full list of supported optimization passes can be found using get_available_passes()
    all_passes = optimizer.get_available_passes()
    print("Available optimization passes:")
    for p in all_passes:
        print(p)
    print()

    # Pick one pass as example
    passes = ['fuse_consecutive_transposes']
    optimized_model = optimizer.optimize(original_model)

    onnx.save(optimized_model, onnx_out_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_type", type=str, default="turn2onnx", help="run type, turn2onnx|optimize") 
    parser.add_argument("--model_path", type=str, default="test.pth", help="pytorch model_path") 
    parser.add_argument("--onnx_path", type=str, default="test.onnx", help="onnx_path") 
    parser.add_argument("--onnx_path_optimize", type=str, default="test_optimize.onnx", help="onnx_path_optimize") 
    args = parser.parse_args()

    if args.run_type=="turn2onnx":
        pytorch2onnx(args.model_path, args.onnx_path)
        printnode(args.onnx_path)
    elif args.run_type=="optimize":
        printnode(args.onnx_path)
        optimize(args.onnx_path, args.onnx_path_optimize)
        printnode(args.onnx_path_optimize)
    else:
        print("run_type error")
        exit(0)
