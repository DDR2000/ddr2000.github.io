+++
title = 'An introduction in the context of an early project'
date = 2024-06-11T07:07:07+01:00
draft = false
+++

With the prospect of slowly populating this website with my past and ongoing projects, I figured I'd do a sort of introduction post while I take you through one of my earlier projects, Etel. The idea came to me when I first stumbled upon GPGPU programming during my freshman year, don't ask how that happened before I even touched a Graphics API. I was particularly proud of this idea because I came up with it myself, regardless of whether it already existed somewhere out there.

The project in question is a random terrain generator. In hindsight, there was nothing so "terrain" about it. The idea was to start with a 2D array representing the top view of a patch of terrain. In the beginning all elements are assigned the same value, which represents the height at that point on the x-z plain. A mask array of the same dimensions is then subdivided into a 2x2 grid of cells, where each cell is assigned a new height value in a specified range. Each cell of the mask overlaps a quadrant of the original grid and the new values of the mask are applied to coinciding points from the original grid:

{{< figure src="../gridmask.png" caption="Paint Supremacy" align="center">}}

The new grid now has 4 different regions of slightly varied altitude. This process is repeated by subdividing the mask into progressively smaller cells, resulting in more frequent regions, with peaks and troughs.

The motivation behind the project was taking CUDA for a test drive. Still not confident in my C++ at the time, I picked up PyCUDA to parallelize the mask application over the large 2D array in Python. The rather simple CUDA kernel in Inline C looked like this:

```
kernel_code = """
__global__ void compute_heights(float *a, float *b, float *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    c[i,j] = (a[i,j] + b[i,j])/2;
}
"""
```

It would be much later when I took ML classes that I would realize how useful this can be for training models. Recall how I mentioned earlier how this was less "terrain" and more a random array. Without any visualization, this might as well be a PyCUDA tutorial for nested for loops. Creating this website finally presented an excuse to resurrect this script and feed the terrain array into a WebGL renderer.

<script src="https://greggman.github.io/webgl-lint/webgl-lint.js" crossorigin></script>
<script type="text/javascript" src="../Common/initShaders2.js"></script>
<script type="text/javascript" src="../Common/MVnew.js"></script>
<script type="text/javascript" src="../data.json"></script>
<script type="text/javascript" src="../camera.js"></script>
<script type="text/javascript" src="../Drawable.js"></script>
<script type="text/javascript" src="../square.js"></script>
<script type="text/javascript" src="../app.js"></script>
<script>
console.log(window.location.pathname);
</script>
<div align="center">
<canvas id="gl-canvas" width="700" height="500">
</canvas>
</div>

Good stuff. Going back to this made me realize I could go ham with this. Introduce control points to influence peaks or valleys. Let camera control simulate traversing the terrain with the mesh as a bound, could probably do that right now. But alas, there's more fun to be had. Elsewhere.
<!--more-->
