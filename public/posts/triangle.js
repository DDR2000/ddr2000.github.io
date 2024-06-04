

class Triangle3D extends Drawable{
    static vertexPositions = [
      vec3( 0, 1, 0 ),
    vec3( -0.5, 0, 0.5 ),
    vec3( 0.5, 0, 0.5 ),
    vec3( 0.5, 0, -0.5 ),
    vec3( -0.5, 0, -0.5),
    vec3( -0.5, 0, 0.5 )];

    static vertexPositionsWire = [ //Line loop won't produce a complete wireframe with the same vertices as the positions for the pyramid.
    vec3( -0.5, 0, 0.5 ),
    vec3( 0.5, 0, 0.5 ),
    vec3(0, 1, 0),
    vec3( -0.5, 0, 0.5 ),
    vec3( -0.5, 0, -0.5 ),
    vec3(0, 1, 0),
    vec3( -0.5, 0, -0.5 ),
    vec3( 0.5, 0, -0.5),
    vec3(0, 1, 0),
    vec3( 0.5, 0, 0.5 ),
    vec3( 0.5, 0, -0.5)];

    static vertexColors = [
    	vec3(1,0,0),
    	vec3(0,1,0),
    	vec3(0,0,1),
    	vec3(1,0,1),
    	vec3(0,1,1),
    	vec3(0,1,0)];

    static blend = 1;

    static positionBuffer = -1;
    static shaderProgram = -1;
    static aPositionShader = -1;
    static aColorShader = -1;
    static uCameraMatrixShader = -1;
    static uModelMatrixShader = -1;
    static uProjectionShader = -1;

    static positionBufferWire = -1;
    static shaderProgramLine = -1;
    static aPositionShaderLine = -1;
    static aColorShaderLine = -1;
    static uCameraMatrixShaderLine = -1;
    static uModelMatrixShaderLine = -1;
    static uProjectionShaderLine = -1;



    static initialize() {
    	Triangle3D.shaderProgram = initShaders( gl, "/vshader.glsl", "/fshader.glsl");

    	// Load the data into the GPU
    	Triangle3D.positionBuffer = gl.createBuffer();
    	gl.bindBuffer( gl.ARRAY_BUFFER, Triangle3D.positionBuffer);
    	gl.bufferData( gl.ARRAY_BUFFER, flatten(Triangle3D.vertexPositions), gl.STATIC_DRAW );

      Triangle3D.positionBufferWire = gl.createBuffer();
    	gl.bindBuffer( gl.ARRAY_BUFFER, Triangle3D.positionBufferWire);
    	gl.bufferData( gl.ARRAY_BUFFER, flatten(Triangle3D.vertexPositionsWire), gl.STATIC_DRAW );

      Triangle3D.colorBuffer = gl.createBuffer();
    	gl.bindBuffer( gl.ARRAY_BUFFER, Triangle3D.colorBuffer);
    	gl.bufferData( gl.ARRAY_BUFFER, flatten(Triangle3D.vertexColors), gl.STATIC_DRAW );

    	// Associate our shader variables with our data buffer
    	Triangle3D.aPositionShader = gl.getAttribLocation( Triangle3D.shaderProgram, "aPosition" );
      Triangle3D.aColorShader = gl.getAttribLocation( Triangle3D.shaderProgram, "aColor" );
    	Triangle3D.uModelMatrixShader = gl.getUniformLocation( Triangle3D.shaderProgram, "modelMatrix" );
      Triangle3D.uCameraMatrixShader = gl.getUniformLocation( Triangle3D.shaderProgram, "cameraMatrix");
      Triangle3D.uProjectionShader = gl.getUniformLocation(Triangle3D.shaderProgram, "projectionMatrix");

      Triangle3D.shaderProgramLine = initShaders( gl, "/vshaderUniformColor.glsl", "/fshader.glsl");
    	Triangle3D.aPositionShaderLine = gl.getAttribLocation( Triangle3D.shaderProgramLine, "aPosition" );
    	Triangle3D.uColorShaderLine = gl.getUniformLocation( Triangle3D.shaderProgramLine, "uColor" );
    	Triangle3D.uModelMatrixShaderLine = gl.getUniformLocation( Triangle3D.shaderProgramLine, "modelMatrix" );
      Triangle3D.uCameraMatrixShaderLine = gl.getUniformLocation( Triangle3D.shaderProgramLine, "cameraMatrix");
      Triangle3D.uProjectionShaderLine = gl.getUniformLocation(Triangle3D.shaderProgramLine, "projectionMatrix");
    }

    changeShader(){
      if (Triangle3D.blend==1){
        Triangle3D.shaderProgram = initShaders( gl, "/vshaderflat.glsl", "/fshaderflat.glsl");
        Triangle3D.aPositionShader = gl.getAttribLocation( Triangle3D.shaderProgram, "aPosition" );
        Triangle3D.aColorShader = gl.getAttribLocation( Triangle3D.shaderProgram, "aColor" );
      	Triangle3D.uModelMatrixShader = gl.getUniformLocation( Triangle3D.shaderProgram, "modelMatrix" );
        Triangle3D.uCameraMatrixShader = gl.getUniformLocation( Triangle3D.shaderProgram, "cameraMatrix");
        Triangle3D.uProjectionShader = gl.getUniformLocation(Triangle3D.shaderProgram, "projectionMatrix");
        Triangle3D.blend = -1;
      }
      else{
        Triangle3D.shaderProgram = initShaders( gl, "/vshader.glsl", "/fshader.glsl");
        Triangle3D.aPositionShader = gl.getAttribLocation( Triangle3D.shaderProgram, "aPosition" );
        Triangle3D.aColorShader = gl.getAttribLocation( Triangle3D.shaderProgram, "aColor" );
      	Triangle3D.uModelMatrixShader = gl.getUniformLocation( Triangle3D.shaderProgram, "modelMatrix" );
        Triangle3D.uCameraMatrixShader = gl.getUniformLocation( Triangle3D.shaderProgram, "cameraMatrix");
        Triangle3D.uProjectionShader = gl.getUniformLocation(Triangle3D.shaderProgram, "projectionMatrix");
        Triangle3D.blend = 1;
      }
    }

    constructor(tx, ty, tz, scale, rotX, rotY, rotZ){
        super(tx, ty, tz, scale, rotX, rotY, rotZ);
        if(Triangle3D.shaderProgram == -1)
            Triangle3D.initialize();

        this.modelMatrix = mat4();
        this.modelMatrixWire = mat4()
    }

    draw() {
      gl.useProgram(Triangle3D.shaderProgram);

      gl.bindBuffer( gl.ARRAY_BUFFER, Triangle3D.positionBuffer);
      gl.vertexAttribPointer(Triangle3D.aPositionShader,3, gl.FLOAT,false,0,0 );

      gl.bindBuffer( gl.ARRAY_BUFFER, Triangle3D.colorBuffer);
      gl.vertexAttribPointer(Triangle3D.aColorShader, 3, gl.FLOAT, false, 0, 0 );

      gl.uniformMatrix4fv(Triangle3D.uModelMatrixShader, false, flatten(this.modelMatrix));
      gl.uniformMatrix4fv(Triangle3D.uCameraMatrixShader, false, flatten(cam1.cameraMatrix));
      gl.uniformMatrix4fv(Triangle3D.uProjectionShader, false, flatten(cam1.projectionMatrix));
      gl.enableVertexAttribArray(Triangle3D.aPositionShader);
      gl.enableVertexAttribArray(Triangle3D.aColorShader);
      gl.drawArrays( gl.TRIANGLE_FAN, 0, Triangle3D.vertexPositions.length);
      gl.disableVertexAttribArray(Triangle3D.aPositionShader);
      gl.disableVertexAttribArray(Triangle3D.aColorShader);

      //Line loop
      gl.useProgram(Triangle3D.shaderProgramLine);

      gl.bindBuffer( gl.ARRAY_BUFFER, Triangle3D.positionBufferWire);
      gl.vertexAttribPointer(Triangle3D.aPositionShaderLine,3, gl.FLOAT,false,0,0 );

      gl.uniform3fv(Triangle3D.uColorShaderLine, vec3(0,0,0));

      gl.uniformMatrix4fv(Triangle3D.uModelMatrixShaderLine, false, flatten(this.modelMatrixWire));
      gl.uniformMatrix4fv(Triangle3D.uCameraMatrixShaderLine, false, flatten(cam1.cameraMatrix));
      gl.uniformMatrix4fv(Triangle3D.uProjectionShaderLine, false, flatten(cam1.projectionMatrix));
      gl.enableVertexAttribArray(Triangle3D.aPositionShaderLine);
      gl.drawArrays( gl.LINE_LOOP, 0, Triangle3D.vertexPositionsWire.length);
      gl.disableVertexAttribArray(Triangle3D.aPositionShaderLine);
    }
}
