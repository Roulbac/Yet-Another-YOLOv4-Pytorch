import os
import argparse
import torch
import coremltools as ct
import coremltools.converters.mil.mil.types as types
from model import YOLOv4

# softplus implementation in coremltools_install_path/converters/mil/frontend/torch/ops.py
#
# @register_torch_op
# def softplus(context, node):
#     inputs = _get_inputs(context, node)
#
#     x = inputs[0]
#     res = mb.softplus(x=x, name=node.name)
#     context.add(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=608)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output', type=str, default='yolov4.mlmodel')

    args = parser.parse_args()

    model = YOLOv4(
        n_classes=args.n_classes,
        weights_path=args.weights,
        img_dim=args.img_size
    )
    print('Created YOLOv4 torch model')
    model.eval()

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    traced_model = torch.jit.trace(model, x)
    print('Traced torch model')

    mlmodel_input = ct.TensorType("input_1", x.shape, dtype=types.fp32)
    mlmodel = ct.convert(traced_model, inputs=[mlmodel_input])
    print('Created MLModel')
    mlmodel.save(args.output)
    print('Saved MLModel under {}'.format(args.output))

