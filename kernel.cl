__kernel void convolution(
    const __global float * inputImage,
    const __global float * filter,
    __global float * outputImage,
    __global const int imageWidth,
    __global const int imageHeight,
    __global const int filterWidth)
{
    // const int id = get_global_id(0);
    int i = get_global_id(0); // horizontal
    int j = get_global_id(1); // vertical
    // printf("[%d][%d]\n", j, i);
    int halffilterSize = filterWidth / 2;
    float sum = 0;
    // printf("halffilterSize:%d\n", halffilterSize);
    for (int l = -halffilterSize; l <= halffilterSize; l++) // vertical
    {
        // const int idxIntmp = (j + l) * imageWidth + i;
        for (int k = -halffilterSize; k <= halffilterSize; k++) // horizontal
        {
            if (i + k >= 0 && i + k < imageWidth && j + l >= 0 && j + l < imageHeight)
            {
                sum += inputImage[(j + l) * imageWidth + i + k] * filter[(l + halffilterSize) * filterWidth + k + halffilterSize];
                    // printf("inputImage[(j + l) * imageWidth + i + k]:%f\n", inputImage[(j + l) * imageWidth + i + k]);
            }
        }
    }
    outputImage[j * imageWidth + i] = sum;
    // printf("global_id:%d output[%d][%d]:%d\n", id, j, i, output[j * imageWidth + i]);
}
