+++
title = 'Sketch1 Post'
date = 2024-07-01T09:12:30-04:00
draft = false
+++

To absolutely no one's surprise I didn't actually go through with updating this page regularly. The good news is I've been keeping busy with projects.

One of those projects is what I'm very creatively calling SketchTo3D. The idea is fairly simple, but we'll talk about the main motivation behind this project in a bit. Let's first look at an example input.

{{< figure src="../sketch.jpg" align="center" width="250" >}}

The goal is to turn drawings like this one into a 3D mesh, while somehow retaining depth information. It may or may not be apparent that the intention with this drawing was to have the bigger square be closer than the smaller square in the back. The algorithm needs to somehow read this intention and also derive the depth in model space.

The first step seems simple, we need edge information and/or corner information. However, edge detecting the raw source gives us some chaotic results.

{{< figure src="../edges_unfiltered.png" align="center" width="400" >}}

OpenCV provides a nifty corner detection utility with goodFeaturesToTrack() but it's going to return every intersection with the rough lines. There are several ways to clean up the edges but none of the ones I thought of initially were good solutions. Simple discarding the shorter lines over some region between corner candidates, generalizing by gradient, could work but might fall apart over other samples. The work around then has to somehow convert human sketches to lines with less character.

{{< figure src="../edges_unfiltered2.png" align="center" width="400" >}}

By using thicker lines, the problem is reframed from detecting and discarding noise to simply thinning one thick line for every edge. I think a package of OpenCV also includes a thinning utility but the [theory for Zhang-Suen thinning](https://dl.acm.org/doi/pdf/10.1145/357994.358023) seems simple enough so lets implement it.

```
def conditionA(img, y, x):
    x0, y0, x1, y1 = x-1, y-1, x+1, y+1
    #[p2,p3,p4,p5,p6,p7,p8,p9]
    neighbours=[img[y0,x],img[y0,x1],img[y,x1],img[y1,x1],img[y1,x],img[y1,x0],img[y,x0],img[y0,x0]]
    transitions=0
    for i in range(1, len(neighbours)):
        transitions += (neighbours[i]>neighbours[i-1])*(neighbours[i]-neighbours[i-1])
    transitions += (neighbours[0]>neighbours[-1])*(neighbours[0]-neighbours[-1])
    return transitions

def conditionB(img, y, x):
    x0, y0, x1, y1 = x-1, y-1, x+1, y+1
    return img[y0,x] + img[y0,x1] + img[y,x1] + img[y1,x1] + img[y1,x] + img[y1,x0] + img[y,x0] + img[y0,x0]

def thinning(img, rMin, rMax):
    img = np.where(img>127, 0, 1)
    while(True):
        note1=[]
        for i in range(rMin[0],rMax[0]-1):
            for j in range(rMin[1], rMax[1]-1):
                A = conditionA(img, j, i)
                B = conditionB(img, j, i)
                c = img[j,i] and (B >= 2) and (B <= 6) and (A == 1) and ((img[j-1, i]==0) or (img[j, i+1]==0) or (img[j+1, i]==0)) and ((img[j, i+1]==0) or (img[j+1, i]==0) or (img[j, i-1])==0)
                if c:
                    note1.append([j,i])
        for p in note1:
            img[p[0],p[1]]=0
        note2 = []
        for i in range(rMin[0],rMax[0]-1):
            for j in range(rMin[1], rMax[1]-1):
                A = conditionA(img, j, i)
                B = conditionB(img, j, i)
                c = img[j,i] and (B >= 2) and (B <= 6) and (A == 1) and ((img[j-1, i]==0) or (img[j, i+1]==0) or (img[j, i-1]==0)) and ((img[j-1, i]==0) or (img[j+1, i]==0) or (img[j, i-1])==0)
                if c:
                    note2.append([j,i])
        for p in note2:
            img[p[0],p[1]]=0
        if(len(note1)==0 or len(note2)==0):
            break

    white = (0,0,0)
    black = (255,255,255)

    res = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Map binary values to the respective colors
    res[img == 0] = white
    res[img == 1] = black

    img = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    return img
```

Binarizing the image before passing it onto thinning produces a much cleaner image. Here are the detected points superimposed on the thinned image.

{{< figure src="../res_points.jpg" align="center" width="600" >}}

So now let's actually address the main motivation I talked about earlier. In a previous research project, I looked at the viability of using AI images to compose 3D scenes, sort of bypassing the immense compute required to train a model to generate believable 3D scenes directly. One of the pain points of that project was mapping *believable* depths to each object. A depth mask from some existing model does provide relative depth but how about in this case? This project isolates that problem more so I can think about how best to derive model space depth.

With corner information, I'm only generating a flat shape. If samples are made to always be in perspective, a reliable method of deriving depth can be defining vanishing points, and in turn the x-y plane. A z-vector for candidate lines then becomes a simple affair.

Thanks for reading thus far, I'll update this series with another post after I test some more ideas.
