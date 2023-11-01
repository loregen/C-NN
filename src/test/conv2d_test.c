#include "math.h"
#include "net.h"

int main(void){

    Conv2DLayer conv2d = *Conv2DLayer_init(3, 3, 1, 3, 2, 2);

    Tensor input = Tensor_init(4, SHAPE(2, 1, 3, 3), true, false);
    for(size_t i = 0; i < input.size; i++){
        input.data[i] = i;
    }

    conv2d.base.input = input;
    conv2d.base.forward_init(&conv2d.base, 2);

    conv2d.base.check_shapes(&conv2d.base);

    Conv2DLayer_forward(&conv2d.base);

    Tensor_print(&conv2d.base.input, "input", true);
    Tensor_print(&conv2d.kernels, "kernels", true);
    Tensor_print(&conv2d.biases, "biases", true);
    Tensor_print(&conv2d.base.output, "output", true);

    return 0;
}