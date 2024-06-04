async function getJSON() {
    return fetch('../data.json')
        .then((response)=>response.json())
        .then((responseJson)=>{return responseJson});
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

class Square3D extends Drawable{
    static vertexPositions = [];
	static vertexColors = [];
	static heights = [];


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
	
	
	static async computeVertices(){
		Square3D.heights = await getJSON();
		for (let j = 0; j <= 118; j=j+1) {
			for (let i = 0; i <= 118; i=i+1) {
				Square3D.vertexPositions.push(vec3(i/5,Square3D.heights[i][j]/100,j/5))
				Square3D.vertexColors.push(vec3(0.250,Square3D.heights[i][j]/100,0.176))
				Square3D.vertexPositions.push(vec3((i+1)/5,Square3D.heights[i+1][j+1]/100,(j+1)/5))
				Square3D.vertexColors.push(vec3(0.250,Square3D.heights[i+1][j+1]/100,0.176))
				Square3D.vertexPositions.push(vec3(i/5,Square3D.heights[i][j+1]/100,(j+1)/5))
				Square3D.vertexColors.push(vec3(0.250,Square3D.heights[i][j+1]/100,0.176))
				
				Square3D.vertexPositions.push(vec3(i/5,Square3D.heights[i][j]/100,j/5))
				Square3D.vertexColors.push(vec3(0.250,Square3D.heights[i][j]/100,0.176))
				Square3D.vertexPositions.push(vec3((i+1)/5,Square3D.heights[i+1][j+1]/100,(j+1)/5))
				Square3D.vertexColors.push(vec3(0.250,Square3D.heights[i+1][j+1]/100,0.176))
				Square3D.vertexPositions.push(vec3((i+1)/5,Square3D.heights[i+1][j]/100,j/5))
				Square3D.vertexColors.push(vec3(0.250,Square3D.heights[i+1][j]/100,0.176))
			}
		}
		Square3D.vertexPositions.push(vec3(119,Square3D.heights[119][119]/100,199))
		Square3D.vertexColors.push(vec3(0.250,Square3D.heights[119][119]/100,0.176))
		this.vertexPositionsWire = Square3D.vertexPositions;
	}


    static async initialize() {
		Square3D.computeVertices();
		await sleep(1000);
    	Square3D.shaderProgram = initShaders( gl, "../vshader.glsl", "../fshader.glsl");

    	// Load the data into the GPU
    	Square3D.positionBuffer = gl.createBuffer();
    	gl.bindBuffer( gl.ARRAY_BUFFER, Square3D.positionBuffer);
    	gl.bufferData( gl.ARRAY_BUFFER, flatten(Square3D.vertexPositions), gl.STATIC_DRAW );
		
		Square3D.positionBufferWire = gl.createBuffer();
    	gl.bindBuffer( gl.ARRAY_BUFFER, Square3D.positionBufferWire);
    	gl.bufferData( gl.ARRAY_BUFFER, flatten(Square3D.vertexPositionsWire), gl.STATIC_DRAW );

      Square3D.colorBuffer = gl.createBuffer();
    	gl.bindBuffer( gl.ARRAY_BUFFER, Square3D.colorBuffer);
    	gl.bufferData( gl.ARRAY_BUFFER, flatten(Square3D.vertexColors), gl.STATIC_DRAW );

    	// Associate our shader variables with our data buffer
    	Square3D.aPositionShader = gl.getAttribLocation( Square3D.shaderProgram, "aPosition" );
      Square3D.aColorShader = gl.getAttribLocation( Square3D.shaderProgram, "aColor" );
    	Square3D.uModelMatrixShader = gl.getUniformLocation( Square3D.shaderProgram, "modelMatrix" );
      Square3D.uCameraMatrixShader = gl.getUniformLocation( Square3D.shaderProgram, "cameraMatrix");
      Square3D.uProjectionShader = gl.getUniformLocation(Square3D.shaderProgram, "projectionMatrix");
	  
	  Square3D.shaderProgramLine = initShaders( gl, "../vshaderUniformColor.glsl", "../fshader.glsl");
    	Square3D.aPositionShaderLine = gl.getAttribLocation( Square3D.shaderProgramLine, "aPosition" );
    	Square3D.uColorShaderLine = gl.getUniformLocation( Square3D.shaderProgramLine, "uColor" );
    	Square3D.uModelMatrixShaderLine = gl.getUniformLocation( Square3D.shaderProgramLine, "modelMatrix" );
      Square3D.uCameraMatrixShaderLine = gl.getUniformLocation( Square3D.shaderProgramLine, "cameraMatrix");
      Square3D.uProjectionShaderLine = gl.getUniformLocation(Square3D.shaderProgramLine, "projectionMatrix");
    }

    constructor(tx, ty, tz, scale, rotX, rotY, rotZ){
        super(tx, ty, tz, scale, rotX, rotY, rotZ);
        if(Square3D.shaderProgram == -1)
            Square3D.initialize();

        this.modelMatrix = mat4();
		this.modelMatrixWire = mat4();
    }

    async draw() {
      gl.useProgram(Square3D.shaderProgram);

      gl.bindBuffer( gl.ARRAY_BUFFER, Square3D.positionBuffer);
      gl.vertexAttribPointer(Square3D.aPositionShader,3, gl.FLOAT,false,0,0 );

      gl.bindBuffer( gl.ARRAY_BUFFER, Square3D.colorBuffer);
      gl.vertexAttribPointer(Square3D.aColorShader, 3, gl.FLOAT, false, 0, 0 );

      gl.uniformMatrix4fv(Square3D.uModelMatrixShader, false, flatten(this.modelMatrix));
      gl.uniformMatrix4fv(Square3D.uCameraMatrixShader, false, flatten(cam1.cameraMatrix));
      gl.uniformMatrix4fv(Square3D.uProjectionShader, false, flatten(cam1.projectionMatrix));
      gl.enableVertexAttribArray(Square3D.aPositionShader);
      gl.enableVertexAttribArray(Square3D.aColorShader);
      gl.drawArrays( gl.TRIANGLES, 0, Square3D.vertexPositions.length);
      gl.disableVertexAttribArray(Square3D.aPositionShader);
      gl.disableVertexAttribArray(Square3D.aColorShader);
	  
	  //Line loop
      gl.useProgram(Square3D.shaderProgramLine);

      gl.bindBuffer( gl.ARRAY_BUFFER, Square3D.positionBufferWire);
      gl.vertexAttribPointer(Square3D.aPositionShaderLine,3, gl.FLOAT,false,0,0 );

      gl.uniform3fv(Square3D.uColorShaderLine, vec3(0,0,0));

      gl.uniformMatrix4fv(Square3D.uModelMatrixShaderLine, false, flatten(this.modelMatrixWire));
      gl.uniformMatrix4fv(Square3D.uCameraMatrixShaderLine, false, flatten(cam1.cameraMatrix));
      gl.uniformMatrix4fv(Square3D.uProjectionShaderLine, false, flatten(cam1.projectionMatrix));
      gl.enableVertexAttribArray(Square3D.aPositionShaderLine);
      gl.drawArrays( gl.LINES, 0, Square3D.vertexPositionsWire.length);
      gl.disableVertexAttribArray(Square3D.aPositionShaderLine);
    }
}
