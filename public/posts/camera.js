class Camera{

  constructor(vrp,u,v,n){
    this.vrp = vrp;
    this.u = normalize(u);
    this.v = normalize(v);
    this.n = normalize(n);
    this.projectionMatrix = perspective(45.0, 1.0, 0.1, 100);
    this.updateCameraMatrix();
  }

  updateCameraMatrix(){
    let t = translate(-this.vrp[0],-this.vrp[1],-this.vrp[2]);
    let r = mat4(this.u[0], this.u[1], this.u[2], 0,
    this.v[0], this.v[1], this.v[2], 0,
    this.n[0], this.n[1], this.n[2], 0,
    0, 0, 0, 1);
    this.cameraMatrix = mult(r,t);
  }

}
