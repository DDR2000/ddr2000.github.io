#version 300 es
in vec3 aPosition;
out vec4 vColor;

uniform vec3 uColor;
uniform mat4 modelMatrix;
uniform mat4 cameraMatrix;
uniform mat4 projectionMatrix;

void main()
{
    gl_Position = projectionMatrix*cameraMatrix*modelMatrix*vec4(aPosition,1.0); //cahnge this to make wire+triangle

    vColor = vec4(uColor,1.0);
}
