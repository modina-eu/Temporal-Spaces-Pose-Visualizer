

'use strict';


const canvas = document.getElementsByTagName('canvas')[0];
resizeCanvas();

let config = {
    DYE_RESOLUTION: 512,
    PAUSED: false,
    BACK_COLOR: { r: 0, g: 0, b: 0 },
    TRANSPARENT: false,
  //  SUNRAYS: true,
  //  SUNRAYS_RESOLUTION: 1024,
}

function pointerPrototype () {
    this.id = -1;
    this.texcoordX = 0;
    this.texcoordY = 0;
    this.prevTexcoordX = 0;
    this.prevTexcoordY = 0;
    this.deltaX = 0;
    this.deltaY = 0;
    this.down = false;
    this.moved = false;
}

let pointers = [];
let splatStack = [];
let s1 = 0;
let s2 = 0;
let s3 = 0.5;
let s4 = 0;
let s5 = 0;
let p1 = 0;
pointers.push(new pointerPrototype());

const { gl, ext } = getWebGLContext(canvas);

if (isMobile()) {
    //config.DYE_RESOLUTION = 512;
  //  config.SUNRAYS_RESOLUTION = 512;
}
if (!ext.supportLinearFiltering) {
  //  config.DYE_RESOLUTION = 512;
//  config.SUNRAYS_RESOLUTION = 512;

}

function getWebGLContext (canvas) {
    const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };

    let gl = canvas.getContext('webgl2', params);
    const isWebGL2 = !!gl;
    if (!isWebGL2)
        gl = canvas.getContext('webgl', params) || canvas.getContext('experimental-webgl', params);

    let halfFloat;
    let supportLinearFiltering;
    if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float');
        supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
    } else {
        halfFloat = gl.getExtension('OES_texture_half_float');
        supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
    let formatRGBA;
    let formatRG;
    let formatR;

    if (isWebGL2)
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);
    }
    else
    {
        formatRGBA = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.RGBA, gl.RGBA, halfFloatTexType);
    }

    ga('send', 'event', isWebGL2 ? 'webgl2' : 'webgl', formatRGBA == null ? 'not supported' : 'supported');

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
            supportLinearFiltering
        }
    };
}

function getSupportedFormat (gl, internalFormat, format, type)
{
    if (!supportRenderTextureFormat(gl, internalFormat, format, type))
    {
        switch (internalFormat)
        {
            case gl.R16F:
                return getSupportedFormat(gl, gl.RG16F, gl.RG, type);
            case gl.RG16F:
                return getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, type);
            default:
                return null;
        }
    }

    return {
        internalFormat,
        format
    }
}

function supportRenderTextureFormat (gl, internalFormat, format, type) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    let fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);

    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    return status == gl.FRAMEBUFFER_COMPLETE;
}


function isMobile () {
    return /Mobi|Android/i.test(navigator.userAgent);
}

function framebufferToTexture (target) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    let length = target.width * target.height * 4;
    let texture = new Float32Array(length);
    gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.FLOAT, texture);
    return texture;
}


class Material {
    constructor (vertexShader, fragmentShaderSource) {
        this.vertexShader = vertexShader;
        this.fragmentShaderSource = fragmentShaderSource;
        this.programs = [];
        this.activeProgram = null;
        this.uniforms = [];
    }

    setKeywords (keywords) {
        let hash = 0;
        for (let i = 0; i < keywords.length; i++)
            hash += hashCode(keywords[i]);

        let program = this.programs[hash];
        if (program == null)
        {
            let fragmentShader = compileShader(gl.FRAGMENT_SHADER, this.fragmentShaderSource, keywords);
            program = createProgram(this.vertexShader, fragmentShader);
            this.programs[hash] = program;
        }

        if (program == this.activeProgram) return;

        this.uniforms = getUniforms(program);
        this.activeProgram = program;
    }

    bind () {
        gl.useProgram(this.activeProgram);
    }
}

class Program {
    constructor (vertexShader, fragmentShader) {
        this.uniforms = {};
        this.program = createProgram(vertexShader, fragmentShader);
        this.uniforms = getUniforms(this.program);
    }

    bind () {
        gl.useProgram(this.program);
    }
}

function createProgram (vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.trace(gl.getProgramInfoLog(program));

    return program;
}

function getUniforms (program) {
    let uniforms = [];
    let uniformCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < uniformCount; i++) {
        let uniformName = gl.getActiveUniform(program, i).name;
        uniforms[uniformName] = gl.getUniformLocation(program, uniformName);
    }
    return uniforms;
}

function compileShader (type, source, keywords) {
    source = addKeywords(source, keywords);

    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        console.trace(gl.getShaderInfoLog(shader));

    return shader;
};

function addKeywords (source, keywords) {
    if (keywords == null) return source;
    let keywordsString = '';
    keywords.forEach(keyword => {
        keywordsString += '#define ' + keyword + '\n';
    });
    return keywordsString + source;
}

const baseVertexShader = compileShader(gl.VERTEX_SHADER, `
    precision highp float;

    attribute vec2 aPosition;
    varying vec2 vUv;
    varying vec2 vL;
    varying vec2 vR;
    varying vec2 vT;
    varying vec2 vB;
    uniform vec2 texelSize;

    void main () {
        vUv = aPosition * 0.5 + 0.5;
        //vUv = vec2(a.x,1.-a.y);
        /*vL = vUv - vec2(texelSize.x, 0.0);
        vR = vUv + vec2(texelSize.x, 0.0);
        vT = vUv + vec2(0.0, texelSize.y);
        vB = vUv - vec2(0.0, texelSize.y);*/
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`);

const displayShaderSource = `
    precision highp float;
    precision highp sampler2D;
    varying vec2 vUv;
    uniform sampler2D uTex;
    uniform vec2 resolution;
    uniform float ti;
    uniform float s1;
    uniform float s2;
    uniform float s3;
    uniform float s4;
    uniform float s5;
    float li(vec2 u, vec2 a, vec2 b) { vec2 ua = u - a; vec2 ba = b - a; float h = clamp(dot(ua, ba) / dot(ba, ba), 0., 1.);
			return length(ua - ba * h);
			}
      vec2 tc(float _ti, float _resx2, float v, float itn, float it) {
      return vec2(fract(_ti / _resx2), 1.-(fract(_ti / _resx2 / itn) / it+v/it));
    }
    float map(float value, float min1, float max1, float min2, float max2) {
				return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
			}
    //  uniform vec2 point;
    void main () {
      vec2 uv =  vUv*(1.-s1);
      uv += vec2(s2,s3);
      float ti = ti * 60.*s4;
			float _resx2 = 64.;
			float itn = floor(_resx2 / 12.);
			float it = _resx2 / itn;
			vec2 u0 = tc(ti , _resx2,0. ,itn,it);
			vec2 u1 = tc(ti, _resx2, 1., itn, it);
			vec2 u2 = tc(ti, _resx2, 2., itn, it);
			vec2 u3 = tc(ti, _resx2, 3., itn, it);
			vec2 u4 = tc(ti, _resx2, 4., itn, it);
			vec2 u5 = tc(ti, _resx2, 5., itn, it);
			vec2 u6 = tc(ti, _resx2, 6., itn, it);
			vec2 u7 = tc(ti, _resx2, 7., itn, it);
			vec2 u8 = tc(ti, _resx2, 8., itn, it);
			vec2 u9 = tc(ti, _resx2, 9., itn, it);
			vec2 u10 = tc(ti, _resx2, 10., itn, it);
			vec2 u11 = tc(ti, _resx2, 11., itn, it);

			vec2 ub0 = tc(ti+1., _resx2, 0., itn, it);
			vec2 ub1 = tc(ti + 1., _resx2, 1., itn, it);
			vec2 ub2 = tc(ti + 1., _resx2, 2., itn, it);
			vec2 ub3 = tc(ti + 1., _resx2, 3., itn, it);
			vec2 ub4 = tc(ti + 1., _resx2, 4., itn, it);
			vec2 ub5 = tc(ti + 1., _resx2, 5., itn, it);
			vec2 ub6 = tc(ti + 1., _resx2, 6., itn, it);
			vec2 ub7 = tc(ti + 1., _resx2, 7., itn, it);
			vec2 ub8 = tc(ti + 1., _resx2, 8., itn, it);
			vec2 ub9 = tc(ti + 1., _resx2, 9., itn, it);
			vec2 ub10 = tc(ti + 1., _resx2, 10., itn, it);
			vec2 ub11 = tc(ti + 1., _resx2, 11., itn, it);
			float a = 0.5;
			float ta = smoothstep(0., 1., fract(ti));
			vec2 tp = mix(texture2D(uTex, u7).xy, texture2D(uTex, ub7).xy, ta);
			vec2 p1 = ((mix(texture2D(uTex, u0).xy,texture2D(uTex, ub0).xy,ta) - a) + tp);
			vec2 p2 = ((mix(texture2D(uTex, u1).xy, texture2D(uTex, ub1).xy, ta) - a) + tp);
			vec2 p3 = ((mix(texture2D(uTex, u2).xy, texture2D(uTex, ub2).xy, ta) - a) + tp);
			vec2 p4 = ((mix(texture2D(uTex, u3).xy, texture2D(uTex, ub3).xy, ta) - a) + tp);
			vec2 p5 = ((mix(texture2D(uTex, u4).xy, texture2D(uTex, ub4).xy, ta) - a) + tp);
			vec2 p6 = ((mix(texture2D(uTex, u5).xy, texture2D(uTex, ub5).xy, ta) - a) + tp);
			vec2 p7 = ((mix(texture2D(uTex, u6).xy, texture2D(uTex, ub6).xy, ta) - a) + tp);
			vec2 p8 = ((mix(texture2D(uTex, u8).xy, texture2D(uTex, ub8).xy, ta) - a) + tp);
			vec2 p9 = ((mix(texture2D(uTex, u9).xy, texture2D(uTex, ub9).xy, ta) - a) + tp);
			vec2 p10 = ((mix(texture2D(uTex, u10).xy, texture2D(uTex, ub10).xy, ta) - a) + tp);
			vec2 p11 = ((mix(texture2D(uTex, u11).xy, texture2D(uTex, ub11).xy, ta) - a) + tp);
			float ts = 0.0015/clamp(map(s1,0.3,1.,1.,2.),0.,1.);
      float ts2 = 0.0001/clamp(map(s1,0.3,1.,1.,2.),0.,1.);
			float p = smoothstep(ts,ts2,li(uv, p1, tp));
			p += smoothstep(ts, ts2, li(uv, p2, tp));
			p += smoothstep(ts, ts2, li(uv, p3, tp));
			p += smoothstep(ts, ts2, li(uv, p4, tp));
			p += smoothstep(ts, ts2, li(uv, p5, tp));
			p += smoothstep(ts, ts2, li(uv, p6, tp));
			p += smoothstep(ts, ts2, li(uv, p7, tp));
			p += smoothstep(ts, ts2, li(uv, p8, tp));
			p += smoothstep(ts, ts2, li(uv, p9, tp));
			p += smoothstep(ts, ts2, li(uv, p10, tp));
			p += smoothstep(ts, ts2, li(uv, p11, tp));

			float c = smoothstep(ts, ts2, li(uv, p1, tp ));
			c += smoothstep(ts, ts2, li(uv, p2, p3 ));
			c += smoothstep(ts, ts2, li(uv, p2, p4 ));
			c += smoothstep(ts, ts2, li(uv, p4, p6 ));
			c += smoothstep(ts, ts2, li(uv, p3, p5 ));
			c += smoothstep(ts, ts2, li(uv, p5, p7 ));
			c += smoothstep(ts, ts2, li(uv, tp, p8 ));
			c += smoothstep(ts, ts2, li(uv, tp, p9 ));
			c += smoothstep(ts, ts2, li(uv, p8, p10 ));
			c += smoothstep(ts, ts2, li(uv, p9, p11 ));

        gl_FragColor =  (vec4(0.7,0.,0.,1.)*p+c);
    }
`;


const splatShader = compileShader(gl.FRAGMENT_SHADER, `
  precision highp float;
  precision highp sampler2D;

  varying vec2 vUv;
  uniform sampler2D uTarget;
  uniform float aspectRatio;
  uniform vec2 point;

  void main () {
      vec2 p = vUv - point.xy;
      p.x *= aspectRatio;
      vec3 diff = vec3(ts22*vec2(1.,aspectRatio),0.);
      float mp =smoothstep(0.09,0.,length(p));
      vec4 center =texture2D(uTarget, vUv);
  float top = texture2D(uTarget, vUv-diff.zy).x;
  float left = texture2D(uTarget, vUv-diff.xz).x;
  float right = texture2D(uTarget, vUv+diff.xz).x;
  float bottom = texture2D(uTarget, vUv+diff.zy).x;
  float red = -(center.y-0.5)*2.+(top+left+right+bottom-2.);
  red += mp;red *= 0.99;
  red = 0.5 +red*0.5;
  red = clamp(red,0.,1.);
      gl_FragColor = vec4(0.,0.,0., 1.0);
    }
`);

const blit = (() => {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(0);

    return (target, clear = false) => {
        if (target == null)
        {
            gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        }
        else
        {
            gl.viewport(0, 0, target.width, target.height);
            gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
        }

        gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    }
})();

function CHECK_FRAMEBUFFER_STATUS () {
    let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status != gl.FRAMEBUFFER_COMPLETE)
        console.trace("Framebuffer error: " + status);
}
let fragesTexture = createTextureAsync('capture.png');

const displayMaterial = new Material(baseVertexShader, displayShaderSource);
function initFramebuffers () {

    //let dyeRes = getResolution(config.DYE_RESOLUTION);

    const texType = ext.halfFloatTexType;
    const rgba    = ext.formatRGBA;
    const rg      = ext.formatRG;
    const r       = ext.formatR;
    //const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;
    gl.disable(gl.BLEND);


}
function createTextureAsync (url) {
    let texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255]));

    let obj = {
        texture,
        width: 1,
        height: 1,
        attach (id) {
            gl.activeTexture(gl.TEXTURE0 + id);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            return id;
        }
    };

    let image = new Image();
    image.onload = () => {
        obj.width = image.width;
        obj.height = image.height;
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);
    };
    image.src = url;

    return obj;
}
function updateKeywords () {
    let displayKeywords = [];
    displayMaterial.setKeywords(displayKeywords);
}

updateKeywords();
initFramebuffers();

let lastUpdateti = Date.now();
update();

function update () {
    const dt = calcDeltati();
    if (resizeCanvas())
        initFramebuffers();
    render(null);
    requestAnimationFrame(update);
}
function calcDeltati () {
    let now = Date.now();
    let dt = (now - lastUpdateti) / 1000;
    dt = Math.min(dt, 0.016666);
    lastUpdateti = now;
    return dt;
}
function resizeCanvas () {
    let width = scaleByPixelRatio(canvas.clientWidth);
    let height = scaleByPixelRatio(canvas.clientHeight);
    if (canvas.width != width || canvas.height != height) {
        canvas.width = width;
        canvas.height = height;
        return true;
    }
    return false;
}
function render (target) {

    drawDisplay(target);
}
function drawDisplay (target) {
    let width = target == null ? gl.drawingBufferWidth : target.width;
    let height = target == null ? gl.drawingBufferHeight : target.height;

    displayMaterial.bind();
    gl.uniform1f(displayMaterial.uniforms.ti, performance.now() / 1000);
  gl.uniform2f(displayMaterial.uniforms.resolution, canvas.width , canvas.height);
  gl.uniform1i(displayMaterial.uniforms.uTex, fragesTexture.attach(1));
  gl.uniform1f(displayMaterial.uniforms.s1,s1);
  gl.uniform1f(displayMaterial.uniforms.s2,s2);
  gl.uniform1f(displayMaterial.uniforms.s3,s3);
  gl.uniform1f(displayMaterial.uniforms.s4,s4);
  gl.uniform1f(displayMaterial.uniforms.s5,s5);
    blit(target);
}
function gene(value, sliderId) {
  switch (sliderId) {
    case 'slider1':
      document.getElementById("slider1-value").textContent = value.toFixed(2);
      s1 = value;
      break;
    case 'slider2':
    document.getElementById("slider2-value").textContent = value.toFixed(2);
      s2 = value;
      break;
    case 'slider3':
    document.getElementById("slider3-value").textContent = value.toFixed(2);
      s3 = value;
      break;
      case 'slider4':
      document.getElementById("slider4-value").textContent = value.toFixed(2);
        s4 = value;
        break;
      case 'slider5':
      document.getElementById("slider5-value").textContent = value.toFixed(2);
        s5 = value;
        break;

    default:
      break;
  }
}
function gene2 (value2){
  p1 = value2;
}
function getResolution (resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1)
        aspectRatio = 1.0 / aspectRatio;

    let min = Math.round(resolution);
    let max = Math.round(resolution * aspectRatio);

    if (gl.drawingBufferWidth > gl.drawingBufferHeight)
        return { width: max, height: min };
    else
        return { width: min, height: max };
}

function getTextureScale (texture, width, height) {
    return {
        x: width / texture.width,
        y: height / texture.height
    };
}

function scaleByPixelRatio (input) {
    let pixelRatio = window.devicePixelRatio || 1;
    return Math.floor(input * pixelRatio);
}
