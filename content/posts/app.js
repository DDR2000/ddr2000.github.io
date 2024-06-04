var canvas;
var gl;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

camPos = vec3(0,5,5);
lookAt = vec3(10,0,10);
n = normalize(subtract(camPos, lookAt));
u = cross(vec3(0,1,0),n);
v = cross(n, u);

var cam1 = new Camera(camPos, u, v, n);
window.onload = async function init(){
    canvas = document.getElementById( "gl-canvas" );
    gl = canvas.getContext('webgl2');
    if ( !gl ) { alert( "WebGL 2.0 isn't available" ); }

    gl.viewport( 0, 0, canvas.width, canvas.height );
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor( 0.0, 0.0, 0.0, 1.0 );

    s = new Square3D();
	await sleep(2000);
	theta = 0
    window.addEventListener("keydown",toggle);
    render();
};

async function toggle(){
  switch(event.code){
    case "KeyW":
      cam1.vrp[1] = cam1.vrp[1]-1;
      cam1.vrp[2] = cam1.vrp[2]-1;
      cam1.updateCameraMatrix();
      break;
    case "KeyS":
      cam1.vrp[1] = cam1.vrp[1]+1;
      cam1.vrp[2] = cam1.vrp[2]+1;
      cam1.updateCameraMatrix();
      break;
    case "KeyA":
      cam1.vrp[0] = cam1.vrp[0]-1;
      cam1.updateCameraMatrix();
      break;
    case "KeyD":
      cam1.vrp[0] = cam1.vrp[0]+1;
      cam1.updateCameraMatrix();
      break;
    case "Space":
      t.changeShader();
      break;
  }
}

async function render(){
  setTimeout(function(){
    requestAnimationFrame(render);
    gl.clear(gl.COLOR_BUFFER_BIT);
	theta = theta + (Math.PI/180);
	
	cam1.vrp = [5*Math.cos(theta), 5, 5*Math.sin(theta)];
	cam1.n = normalize(subtract(vec3(cam1.vrp[0], cam1.vrp[1], cam1.vrp[2]), lookAt));
	cam1.u = cross(vec3(0,1,0),cam1.n);
	cam1.v = cross(cam1.n, cam1.u);
	cam1.updateCameraMatrix();
    s.draw();
  }, 30);
}
