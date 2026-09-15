// Local anisotropic 3D Gaussian rasterizer. No external libraries, CDN or CUDA.
// Sigma3 = R diag(s^2) R^T; Sigma2 = J V Sigma3 V^T J^T + 0.3 I.
// Eigenvectors span a 3-sigma quad; fragment alpha is o*exp(-r^2/2).
// A worker depth-sorts instances back-to-front before alpha compositing.
const VS = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec2 corner;
layout(location=1) in uint index;
uniform sampler2D gaussians;
uniform mat4 view;
uniform mat4 projection;
uniform vec2 viewport;
uniform float focal;
uniform float splatScale;
out vec2 local;
out vec4 rgba;
vec4 getData(int i){ int w=textureSize(gaussians,0).x; return texelFetch(gaussians,ivec2(i%w,i/w),0); }
void main(){
 int base=int(index)*4;
 vec4 pos=getData(base), scale=getData(base+1), q=getData(base+2), col=getData(base+3);
 vec4 cam=view*vec4(pos.xyz,1.0);
 local=corner*3.0; rgba=vec4(col.rgb,pos.w);
 if(cam.z>=-0.01){gl_Position=vec4(2.0,2.0,2.0,1.0);rgba.a=0.0;return;}
 float w=q.x,x=q.y,y=q.z,z=q.w;
 mat3 r=mat3(1.0-2.0*(y*y+z*z),2.0*(x*y+w*z),2.0*(x*z-w*y),
             2.0*(x*y-w*z),1.0-2.0*(x*x+z*z),2.0*(y*z+w*x),
             2.0*(x*z+w*y),2.0*(y*z-w*x),1.0-2.0*(x*x+y*y));
 mat3 rs=mat3(view)*r*mat3(scale.x*splatScale,0,0,0,scale.y*splatScale,0,0,0,scale.z*splatScale);
 mat3 sigma=rs*transpose(rs);
 float inv=1.0/(-cam.z);
 vec3 jx=vec3(focal*inv,0.0,focal*cam.x*inv*inv);
 vec3 jy=vec3(0.0,focal*inv,focal*cam.y*inv*inv);
 float a=dot(jx,sigma*jx)+0.3,b=dot(jx,sigma*jy),c=dot(jy,sigma*jy)+0.3;
 float mid=0.5*(a+c), disc=length(vec2(0.5*(a-c),b));
 float l1=max(mid+disc,0.1),l2=max(mid-disc,0.1);
 vec2 dir=abs(b)>0.00001?normalize(vec2(b,l1-a)):(a>=c?vec2(1,0):vec2(0,1));
 vec2 axis1=dir*min(sqrt(l1),700.0),axis2=vec2(-dir.y,dir.x)*min(sqrt(l2),700.0);
 vec4 clip=projection*cam;
 clip.xy+=(axis1*local.x+axis2*local.y)*2.0/viewport*clip.w;
 gl_Position=clip;
}`;
const FS = `#version 300 es
precision highp float;
in vec2 local;
in vec4 rgba;
out vec4 outputColor;
void main(){float r2=dot(local,local);if(r2>9.0)discard;float a=min(0.99,rgba.a*exp(-0.5*r2));if(a<0.0039)discard;outputColor=vec4(rgba.rgb*a,a);}`;

const norm = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
function lookAt(eye, target) { const z = norm(eye.map((v, i) => v - target[i])), x = norm(cross([0, 1, 0], z)), y = cross(z, x); return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]); }
function perspective(fov, aspect) { const f = 1 / Math.tan(fov / 2), near = 0.01, far = 10000; return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) / (near - far), -1, 0, 0, 2 * far * near / (near - far), 0]); }

export class GaussianViewer {
    constructor(canvas, onStats, onError) {
        this.canvas = canvas; this.onStats = onStats; this.onError = onError; this.count = 0; this.scale = 1; this.auto = false; this.autoDirection = 1; this.fov = 50 * Math.PI / 180; this.dirty = true; this.serial = 0; this.lastTime = 0;
        this.init(); this.controls(); this.reset();
        this.observer = new ResizeObserver(() => { this.dirty = true; }); this.observer.observe(canvas);
        canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); this.lost = true; this.onError('The browser lost its graphics context. Reload the page or close other graphics-heavy tabs.'); });
        this.loop = this.loop.bind(this); requestAnimationFrame(this.loop);
    }
    init() {
        const gl = this.canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
        if (!gl) throw Error('WebGL 2 is unavailable. Enable graphics acceleration in Chrome or Edge, then restart the browser.');
        this.gl = gl;
        const shader = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s)); return s; };
        const p = gl.createProgram(); gl.attachShader(p, shader(gl.VERTEX_SHADER, VS)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p));
        this.program = p; gl.useProgram(p);
        this.uniforms = Object.fromEntries(['gaussians', 'view', 'projection', 'viewport', 'focal', 'splatScale'].map(n => [n, gl.getUniformLocation(p, n)]));
        this.vao = gl.createVertexArray(); gl.bindVertexArray(this.vao);
        const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
        this.indices = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.indices); gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0); gl.vertexAttribDivisor(1, 1);
        this.texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.texture); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.clearColor(0.045, 0.072, 0.115, 1);
    }
    load(buffer, meta) {
        const header = new DataView(buffer);
        if (buffer.byteLength < 16 || header.getUint32(0, true) !== 0x31535347 || header.getUint32(8, true) !== 16) throw Error('Unsupported Gaussian scene format.');
        const count = header.getUint32(4, true);
        if (count < 1 || count > 1500000 || buffer.byteLength !== 16 + count * 64) throw Error('The Gaussian scene download is incomplete or too large.');
        this.count = count; this.meta = meta;
        const g = new Float32Array(buffer, 16), gl = this.gl, w = 1024, h = Math.ceil(count * 4 / w), data = new Float32Array(w * h * 4); data.set(g);
        gl.bindTexture(gl.TEXTURE_2D, this.texture); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.indices); gl.bufferData(gl.ARRAY_BUFFER, Uint32Array.from({ length: count }, (_, i) => i), gl.DYNAMIC_DRAW);
        if (this.worker) this.worker.terminate();
        this.sorting = false; this.worker = new Worker('/static/sort-worker.js');
        this.worker.onerror = () => { this.sorting = false; this.onError('Depth sorting stopped. Reload the scene to restore correct transparency.'); };
        this.worker.onmessage = ({ data }) => {
            this.sorting = false; if (data.indices) { gl.bindBuffer(gl.ARRAY_BUFFER, this.indices); gl.bufferData(gl.ARRAY_BUFFER, data.indices, gl.DYNAMIC_DRAW); this.dirty = true; this.sortMs = data.ms; }
            if (data.serial !== this.serial) this.sort();
        };
        const positions = new Float32Array(count * 3); for (let i = 0; i < count; i++)positions.set(g.subarray(i * 16, i * 16 + 3), i * 3);
        this.worker.postMessage({ positions }, [positions.buffer]); this.reset();
    }
    reset() {
        this.target = this.meta?.target?.slice() || [0, 0, -3];
        const eye = this.meta?.source_camera || [0, 0, 0], offset = eye.map((v, i) => v - this.target[i]);
        this.distance = Math.max(Math.hypot(...offset), 0.1); this.yaw = Math.atan2(offset[0], offset[2]); this.pitch = Math.asin(Math.max(-0.999, Math.min(0.999, offset[1] / this.distance)));
        this.baseYaw = this.yaw; this.basePitch = this.pitch; this.viewLimits = this.meta?.view_limits || null; this.autoDirection = 1;
        this.fov = (this.meta?.fov_y || 50) * Math.PI / 180; this.changed();
    }
    preset(name) {
        this.reset(); if (name === 'back') this.yaw += Math.PI; if (name === 'left') this.yaw -= Math.PI / 2; if (name === 'right') this.yaw += Math.PI / 2; if (name === 'top') this.pitch = Math.PI / 2 - 0.01;
        this.changed();
    }
    clampView() {
        if (!this.viewLimits) return;
        const yawLimit = (this.viewLimits.yaw_degrees || 0) * Math.PI / 180, pitchLimit = (this.viewLimits.pitch_degrees || 0) * Math.PI / 180;
        this.yaw = Math.max(this.baseYaw - yawLimit, Math.min(this.baseYaw + yawLimit, this.yaw));
        this.pitch = Math.max(this.basePitch - pitchLimit, Math.min(this.basePitch + pitchLimit, this.pitch));
    }
    changed() { this.clampView(); this.serial++; this.dirty = true; this.view = this.matrix(); this.sort(); }
    matrix() { const cp = Math.cos(this.pitch); this.eye = [this.target[0] + this.distance * cp * Math.sin(this.yaw), this.target[1] + this.distance * Math.sin(this.pitch), this.target[2] + this.distance * cp * Math.cos(this.yaw)]; return lookAt(this.eye, this.target); }
    sort() { if (!this.worker || this.sorting || !this.view) return; this.sorting = true; this.worker.postMessage({ view: Array.from(this.view), serial: this.serial }); }
    controls() {
        const c = this.canvas; let drag = null;
        c.addEventListener('pointerdown', e => { if (drag) return; drag = { id: e.pointerId, x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; c.setPointerCapture(e.pointerId); c.focus(); });
        c.addEventListener('pointermove', e => {
            if (!drag || drag.id !== e.pointerId) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
            if (drag.pan) { const right = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)], up = [-Math.sin(this.pitch) * Math.sin(this.yaw), Math.cos(this.pitch), -Math.sin(this.pitch) * Math.cos(this.yaw)], s = this.distance * 0.0015; for (let i = 0; i < 3; i++)this.target[i] += (-right[i] * dx + up[i] * dy) * s; }
            else { this.yaw -= dx * 0.005; this.pitch = Math.max(-Math.PI / 2 + 0.005, Math.min(Math.PI / 2 - 0.005, this.pitch + dy * 0.005)); }
            this.changed();
        });
        const end = e => { if (drag?.id === e.pointerId) drag = null; }; c.addEventListener('pointerup', end); c.addEventListener('pointercancel', end); c.addEventListener('lostpointercapture', () => drag = null);
        c.addEventListener('contextmenu', e => e.preventDefault());
        c.addEventListener('wheel', e => { e.preventDefault(); this.zoom(Math.exp(Math.max(-100, Math.min(100, e.deltaY)) * 0.002)); }, { passive: false });
        c.addEventListener('keydown', e => { const actions = { ArrowLeft: () => this.yaw -= 0.08, ArrowRight: () => this.yaw += 0.08, ArrowUp: () => this.pitch = Math.min(1.565, this.pitch + 0.08), ArrowDown: () => this.pitch = Math.max(-1.565, this.pitch - 0.08), '+': () => this.zoom(0.9), '=': () => this.zoom(0.9), '-': () => this.zoom(1.1), r: () => this.reset() }; if (actions[e.key]) { e.preventDefault(); actions[e.key](); this.changed(); } });
    }
    zoom(factor) { this.distance = Math.min(10000, Math.max(0.03, this.distance * factor)); this.changed(); }
    loop(now) {
        requestAnimationFrame(this.loop); if (this.lost) return;
        const elapsed = Math.min(0.05, (now - this.lastTime) / 1000); this.lastTime = now;
        if (this.auto && !document.hidden) {
            this.yaw += elapsed * 0.2 * this.autoDirection;
            if (this.viewLimits) {
                const limit = this.viewLimits.yaw_degrees * Math.PI / 180;
                if (this.yaw >= this.baseYaw + limit || this.yaw <= this.baseYaw - limit) this.autoDirection *= -1;
            }
            this.changed();
        }
        if (!this.dirty) return; this.dirty = false;
        const start = performance.now(), gl = this.gl, c = this.canvas, dpr = Math.min(devicePixelRatio || 1, 1.5);
        const width = Math.max(1, Math.floor(c.clientWidth * dpr)), height = Math.max(1, Math.floor(c.clientHeight * dpr));
        if (c.width !== width || c.height !== height) { c.width = width; c.height = height; }
        gl.viewport(0, 0, width, height); gl.clear(gl.COLOR_BUFFER_BIT);
        if (!this.count) return;
        const view = this.view || this.matrix(), projection = perspective(this.fov, width / height), u = this.uniforms;
        gl.useProgram(this.program); gl.bindVertexArray(this.vao); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texture); gl.uniform1i(u.gaussians, 0); gl.uniformMatrix4fv(u.view, false, view); gl.uniformMatrix4fv(u.projection, false, projection); gl.uniform2f(u.viewport, width, height); gl.uniform1f(u.focal, height / (2 * Math.tan(this.fov / 2))); gl.uniform1f(u.splatScale, this.scale); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
        if (now - (this.lastStats || 0) > 500) { this.lastStats = now; this.onStats({ count: this.count, sortMs: this.sortMs || 0, yaw: this.yaw, pitch: this.pitch, drawMs: performance.now() - start }); }
    }
    screenshot() { return new Promise((resolve, reject) => this.canvas.toBlob(b => b ? resolve(b) : reject(Error('Could not capture the viewer.')), 'image/png')); }
}
