class Drawable{
  constructor(tx,ty,tz,scale,rotX, rotY, rotZ){
    this.tx = tx;
    this.ty = ty;
    this.tz = tz;
    this.scale = scale;
    this.modelRotationX = rotX;
    this.modelRotationY = rotY;
    this.modelRotationZ = rotZ;
    this.updateModelMatrix();
  }
  updateModelMatrix(){
    let t = translate(this.tx, this.ty, this.tz);
    let s = scale(this.scale,this.scale,this.scale);
    let rx = rotateX(this.modelRotationX);
    let ry = rotateY(this.modelRotationY);
    let rz = rotateZ(this.modelRotationZ);
    this.modelMatrix = mult(t,mult(s,mult(rz,mult(ry,rx))));
  }
}
