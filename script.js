class CVAE {
    constructor(latent_dim, encoder, decoder) {
        this.latent_dim = latent_dim;
        this.encoder = encoder;
        this.decoder = decoder;
    }

    async sample(eps) {
        if (eps === undefined) {
            eps = tf.randomNormal([100, this.latent_dim]);
        }
        return this.decode(eps, true);
    }

    async encode(x) {
        const [mean, logvar] = tf.split(this.encoder.predict(x), 2, 1);
        return [mean, logvar];
    }

    reparameterize(mean, logvar) {
        const eps = tf.randomNormal(mean.shape);
        return tf.add(tf.mul(tf.exp(tf.mul(logvar, 0.5)), eps), mean);
    }

    async decode(z, apply_sigmoid=false) {
        const logits = this.decoder.predict(z);
        if (apply_sigmoid) {
            return tf.sigmoid(logits);
        }
        return logits;
    }
}
'use strict';
let modelTexture;
let modelTextures;
let tiValue = 0;
let tecreate = 0;
let useModelTexture = false;
const canvas2 = document.getElementById('audio-canvas');
const gl2 = canvas2.getContext('webgl2');
let decoder = null;
let encoder = null;
let generatedImages = [];

async function loadModels() {
    try {
        // Load the encoder and decoder models
        if (!encoder || !decoder) {
          console.log("loding models")
          decoder = await tf.loadGraphModel('model_json/latent_dim_4/augmented_captures/horizontal_flip/250_epochs/decoder/model.json');
          encoder = await tf.loadGraphModel('model_json/latent_dim_4/augmented_captures/horizontal_flip/250_epochs/encoder/model.json');
        }
        // Update the latent dimension
        const latentDim = 4;
        const numExamplesToGenerate = 1; // Change the desired number of examples

        const loaded_model = new CVAE(latentDim, encoder, decoder);

        // Generate a random vector for each example and load the model texture
            const randomVectorForGeneration = tf.randomNormal([1, latentDim]);
            const values = await randomVectorForGeneration.array();
            // vec = [
            //   -0.4324505627155304,
            //   0.8095831871032715,
            //   0.19554875791072845,
            //   0.029588615521788597
            // ]
            vec = [pose.score, pose.score, pose.score, pose.score]
            const newTensor = tf.tensor(values, [1, latentDim]);  // shape must match size
            const sample = await loaded_model.sample(newTensor);
            console.log('sample:',sample)

            modelTexture = await generateImageTexture(loaded_model, sample);
            modelTextures = await generateImageTextures(loaded_model, sample);
            console.log('modelTextures:', modelTextures);

        // Set tecreate to 1 after modelTextures is assigned a value
        tecreate = 1;

        // !! update here the latent dim
        // const latentDim = 4;
        // const numExamplesToGenerate = 1; // Change the desired number of examples
        generatedImages = [];

        for (let i = 0; i < numExamplesToGenerate; i++) {
          const randomVectorForGeneration = tf.randomNormal([1, latentDim]); // Generate a new random vector for each example
          const sample = await loaded_model.sample(randomVectorForGeneration);
          console.log("a");
          const generatedImage = await generateImage(loaded_model, sample);
          console.log(generatedImage);
          generatedImages.push(generatedImage);
        }
        // Display the generated images
        const imageContainer = document.getElementById('imageContainer');
        generatedImages.slice(1);
        generatedImages.forEach((imageData, index) => {
            const canvas = document.createElement('canvas');
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            const context = canvas.getContext('2d');
            context.putImageData(imageData, 0, 0);

            const img = document.createElement('img');
            img.src = canvas.toDataURL();
            img.alt = `Generated Image ${index + 1}`;
            img.classList.add('generatedImage');
            // console.log(img);
            if (imageContainer.hasChildNodes()) {
              imageContainer.removeChild(imageContainer.children[0]);
            }
            imageContainer.appendChild(img);

            fragesTexture = createTextureAsync(imageData);
            fragesTexture = createTextureAsync(imageData);
        });

    } catch (error) {
        console.error('Error loading models:', error);
    }
}
// Generate image texture asynchronously
async function generateImageTexture(model, sample) {
    try {
        const [mean, logvar] = await model.encode(sample);
        const z = model.reparameterize(mean, logvar);
        const prediction = await model.sample(z);
        const pixelsTensor = prediction.squeeze();
        const pixels = await tf.browser.toPixels(pixelsTensor);

        // Create an ImageData object from the pixel data
        const image = new ImageData(new Uint8ClampedArray(pixels), pixelsTensor.shape[1], pixelsTensor.shape[0]);

        // Create a WebGL texture
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.generateMipmap(gl.TEXTURE_2D);

        return texture;
    } catch (error) {
        console.error('Error generating image texture:', error);
        return null;
    }
}
async function generateImageTextures(model, sample) {
    try {
        const [mean, logvar] = await model.encode(sample);
        const z = model.reparameterize(mean, logvar);
        const prediction = await model.sample(z);
        const pixelsTensor = prediction.squeeze();
        const pixels = await tf.browser.toPixels(pixelsTensor);

        // Create an ImageData object from the pixel data
        const image = new ImageData(new Uint8ClampedArray(pixels), pixelsTensor.shape[1], pixelsTensor.shape[0]);

        // Create a WebGL texture
        const texture = gl2.createTexture(); // Change here
        gl2.bindTexture(gl2.TEXTURE_2D, texture); // Change here
        gl2.texParameteri(gl2.TEXTURE_2D, gl2.TEXTURE_MIN_FILTER, gl2.NEAREST); // Set minification filter to nearest neighbor
        gl2.texParameteri(gl2.TEXTURE_2D, gl2.TEXTURE_MAG_FILTER, gl2.NEAREST);
        gl2.texImage2D(gl2.TEXTURE_2D, 0, gl2.RGBA, gl2.RGBA, gl2.UNSIGNED_BYTE, image); // Change here

        //gl2.generateMipmap(gl2.TEXTURE_2D); // Change here

        return texture;
    } catch (error) {
        console.error('Error generating image texture:', error);
        return null;
    }
}
function createTextureAsyncs(url) {
    let texture2 = gl2.createTexture();
    gl2.bindTexture(gl2.TEXTURE_2D, texture2);
    gl2.texParameteri(gl2.TEXTURE_2D, gl2.TEXTURE_MIN_FILTER, gl2.NEAREST);
    gl2.texParameteri(gl2.TEXTURE_2D, gl2.TEXTURE_MAG_FILTER, gl2.NEAREST);
    gl2.texParameteri(gl2.TEXTURE_2D, gl2.TEXTURE_WRAP_S, gl2.CLAMP_TO_EDGE);
    gl2.texParameteri(gl2.TEXTURE_2D, gl2.TEXTURE_WRAP_T, gl2.CLAMP_TO_EDGE);
    gl2.texImage2D(gl2.TEXTURE_2D, 0, gl2.RGB, 1, 1, 0, gl2.RGB, gl2.UNSIGNED_BYTE, new Uint8Array([255, 255, 255]));

    let image = new Image();
    image.onload = () => {
        gl2.bindTexture(gl2.TEXTURE_2D, texture2);
        gl2.texImage2D(gl2.TEXTURE_2D, 0, gl2.RGB, gl2.RGB, gl2.UNSIGNED_BYTE, image);
    };
    image.src = url;
    return texture2;
}
loadModels();

const canvas = document.getElementById('visuals-canvas');
resizeCanvas();
let config = {
    DYE_RESOLUTION: 700,
    PAUSED: false,
    BACK_COLOR: { r: 0, g: 0, b: 0 },
    TRANSPARENT: false,
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
let s1 = 0.;
let s2 = 0.;
let s3 = 0;
let s4 = 0.1;
let s5 = 0;
let p1 = 0;
pointers.push(new pointerPrototype());
const { gl, ext } = getWebGLContext(canvas);
function getWebGLContext (canvas) {
    const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };

    let gl = canvas.getContext('webgl2', params);


    let halfFloat;
    //let supportLinearFiltering;

        gl.getExtension('EXT_color_buffer_float');
        //supportLinearFiltering = gl.getExtension('OES_texture_float_linear');


    gl.clearColor(0.0, 0.0, 0.0, 1.0);

    const halfFloatTexType =  gl.HALF_FLOAT;
    let formatRGBA;
    let formatRG;
    let formatR;

        formatRGBA = getSupportedFormat(gl, gl.RGBA16F, gl.RGBA, halfFloatTexType);
        formatRG = getSupportedFormat(gl, gl.RG16F, gl.RG, halfFloatTexType);
        formatR = getSupportedFormat(gl, gl.R16F, gl.RED, halfFloatTexType);

    ga('send', 'event',  'webgl2' , formatRGBA == null ? 'not supported' : 'supported');

    return {
        gl,
        ext: {
            formatRGBA,
            formatRG,
            formatR,
            halfFloatTexType,
          //  supportLinearFiltering
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
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
`);

const displayShaderSource = `
    precision highp float;
    precision highp sampler2D;
    varying vec2 vUv;
    uniform sampler2D uTex;
    uniform sampler2D uModelTexture;
    uniform vec2 resolution;
    uniform float ti;
    uniform float s1;
    uniform float s2;
    uniform float s3;
    uniform float s4;
    uniform float s5;
    uniform float useModelTexture;
    float li(vec2 u, vec2 a, vec2 b) { vec2 ua = u - a; vec2 ba = b - a; float h = clamp(dot(ua, ba) / dot(ba, ba), 0., 1.);
			return length(ua - ba * h);
			}
      vec2 tc(float _ti, float _resx2, float v, float itn, float it) {
      return vec2(fract(_ti / _resx2), 1.-(fract(_ti / _resx2 / itn) / it+v/it));
    }
    float map(float value, float min1, float max1, float min2, float max2) {
				return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
			}
      float sdl(vec2 p, vec2 lineA, vec2 lineB)
      {
          vec2 lineDir = normalize(lineB - lineA);
          vec2 pointDir = p - lineA;
          float distance = abs(dot(vec2(-lineDir.y, lineDir.x), pointDir));
          return distance;
      }
      float dot2( in vec2 v ) { return dot(v,v); }
      float cro( in vec2 a, in vec2 b ) { return a.x*b.y - a.y*b.x; }
      float bez( vec2 pos, vec2 A,  vec2 B,  vec2 C ){
          vec2 a = B - A;
          vec2 b = A - 2.0*B + C;
          vec2 c = a * 2.0;
          vec2 d = A - pos;
          float kk = 1.0/dot(b,b);
          float kx = kk * dot(a,b);
          float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
          float kz = kk * dot(d,a);
          float res = 0.0;
          float p = ky - kx*kx;
          float p3 = p*p*p;
          float q = kx*(2.0*kx*kx-3.0*ky) + kz;
          float h = q*q + 4.0*p3;
          if( h >= 0.0){
              h = sqrt(h);
              vec2 x = (vec2(h,-h)-q)/2.0;
              float t3 = 1./3.;
              vec2 uv = sign(x)*pow(abs(x), vec2(t3,t3));
              float t = clamp( uv.x+uv.y-kx, 0.0, 1.0 );
              res = dot2(d + (c + b*t)*t);}
          else  {
              float z = sqrt(-p);
              float v = acos( q/(p*z*2.0) ) / 3.0;
              float m = cos(v);
              float n = sin(v)*1.732050808;
              vec3  t = clamp(vec3(m+m,-n-m,n-m)*z-kx,0.0,1.0);
              res = min( dot2(d+(c+b*t.x)*t.x),
                         dot2(d+(c+b*t.y)*t.y) );}
            return sqrt( res );}

            float spo8(vec2 v[8], vec2 p){
    float d = dot(p - v[0], p - v[0]);
    float s = 1.0;
    for (int i = 0; i < 8; i++){
      vec2 ve = v[7];
        if(i>0){ ve = v[i-1];}
      vec2 e = ve - v[i];
              vec2 w =    p - v[i];
              vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
              d = min( d, dot(b,b) );
              bvec3 cond = bvec3( p.y>=v[i].y,
                                  p.y <ve.y,
                                  e.x*w.y>e.y*w.x );
              if( all(cond) || all(not(cond)) ) s=-s;
          }
          return s*sqrt(d);}
          float spo5(vec2 v[5], vec2 p){
  float d = dot(p - v[0], p - v[0]);
  float s = 1.0;
  for (int i = 0; i < 5; i++){
    vec2 ve = v[4];
      if(i>0){ ve = v[i-1];}
    vec2 e = ve - v[i];
            vec2 w =    p - v[i];
            vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
            d = min( d, dot(b,b) );
            bvec3 cond = bvec3( p.y>=v[i].y,
                                p.y <ve.y,
                                e.x*w.y>e.y*w.x );
            if( all(cond) || all(not(cond)) ) s=-s;
        }
        return s*sqrt(d);}
                   float spo4(vec2 v[4], vec2 p){
            float d = dot(p - v[0], p - v[0]);
            float s = 1.0;
            for (int i = 0; i < 4; i++){
              vec2 ve = v[3];
                if(i>0){ ve = v[i-1];}
              vec2 e = ve - v[i];
                      vec2 w =    p - v[i];
                      vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
                      d = min( d, dot(b,b) );
                      bvec3 cond = bvec3( p.y>=v[i].y,
                                          p.y <ve.y,
                                          e.x*w.y>e.y*w.x );
                      if( all(cond) || all(not(cond)) ) s=-s;
                  }
                  return s*sqrt(d);}
                             float spo3(vec2 v[3], vec2 p){
                      float d = dot(p - v[0], p - v[0]);
                      float s = 1.0;
                      for (int i = 0; i < 3; i++){
                        vec2 ve = v[2];
                          if(i>0){ ve = v[i-1];}
                        vec2 e = ve - v[i];
                                vec2 w =    p - v[i];
                                vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
                                d = min( d, dot(b,b) );
                                bvec3 cond = bvec3( p.y>=v[i].y,
                                                    p.y <ve.y,
                                                    e.x*w.y>e.y*w.x );
                                if( all(cond) || all(not(cond)) ) s=-s;
                            }
                            return s*sqrt(d);}
                                       float spo11(vec2 v[11], vec2 p){
                                float d = dot(p - v[0], p - v[0]);
                                float s = 1.0;
                                for (int i = 0; i < 11; i++){
                                  vec2 ve = v[10];
                                    if(i>0){ ve = v[i-1];}
                                  vec2 e = ve - v[i];
                                          vec2 w =    p - v[i];
                                          vec2 b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
                                          d = min( d, dot(b,b) );
                                          bvec3 cond = bvec3( p.y>=v[i].y,
                                                              p.y <ve.y,
                                                              e.x*w.y>e.y*w.x );
                                          if( all(cond) || all(not(cond)) ) s=-s;
                                      }
                                      return s*sqrt(d);}
    vec4 rd4(float t){float ft = floor(t); return fract(sin(vec4(dot(ft,45.236),dot(ft,98.147),dot(ft,23.15),dot(ft,67.19)))*7845.236);}
      float rd(float t){return fract(sin(dot(floor(t),45.236))*7845.236);}
      vec2 tex( float m, vec2 uv ){ if(m<0.5){return texture2D(uModelTexture, uv).xy;}else{return texture2D(uTex, uv).xy;}}
    void main () {
      vec2 uv =  (vUv-0.5)*(1.-s1);
      uv += 0.5;
      uv += vec2(s2,s3);
      uv.x *= 9./16.;
      float ti = ti * 30.*s4;
      float ti2 = floor(ti);
      float ti3 = ti2+1.;
			float _resx2 = 64.;
			float itn = floor(_resx2 / 12.);
			float it = _resx2 / itn;
			vec2 u0 = tc(ti2 , _resx2,0. ,itn,it);
			vec2 u1 = tc(ti2, _resx2, 1., itn, it);
			vec2 u2 = tc(ti2, _resx2, 2., itn, it);
			vec2 u3 = tc(ti2, _resx2, 3., itn, it);
			vec2 u4 = tc(ti2, _resx2, 4., itn, it);
			vec2 u5 = tc(ti2, _resx2, 5., itn, it);
			vec2 u6 = tc(ti2, _resx2, 6., itn, it);
			vec2 u7 = tc(ti2, _resx2, 7., itn, it);
			vec2 u8 = tc(ti2, _resx2, 8., itn, it);
			vec2 u9 = tc(ti2, _resx2, 9., itn, it);
			vec2 u10 = tc(ti2, _resx2, 10., itn, it);
			vec2 u11 = tc(ti2, _resx2, 11., itn, it);

			vec2 ub0 = tc(ti3, _resx2, 0., itn, it);
			vec2 ub1 = tc(ti3, _resx2, 1., itn, it);
			vec2 ub2 = tc(ti3, _resx2, 2., itn, it);
			vec2 ub3 = tc(ti3, _resx2, 3., itn, it);
			vec2 ub4 = tc(ti3, _resx2, 4., itn, it);
			vec2 ub5 = tc(ti3, _resx2, 5., itn, it);
			vec2 ub6 = tc(ti3, _resx2, 6., itn, it);
			vec2 ub7 = tc(ti3, _resx2, 7., itn, it);
			vec2 ub8 = tc(ti3, _resx2, 8., itn, it);
			vec2 ub9 = tc(ti3, _resx2, 9., itn, it);
			vec2 ub10 = tc(ti3, _resx2, 10., itn, it);
			vec2 ub11 = tc(ti3, _resx2, 11., itn, it);
			float a = 0.5;
			float ta = smoothstep(0., 1., fract(ti));
			vec2 tp = mix(tex(useModelTexture, u7), tex(useModelTexture, ub7), ta);
			vec2 p1 = ((mix(tex(useModelTexture, u0),tex(useModelTexture, ub0),ta) - a) + tp);
			vec2 p2 = ((mix(tex(useModelTexture, u1), tex(useModelTexture, ub1), ta) - a) + tp);
			vec2 p3 = ((mix(tex(useModelTexture, u2), tex(useModelTexture, ub2), ta) - a) + tp);
			vec2 p4 = ((mix(tex(useModelTexture, u3), tex(useModelTexture, ub3), ta) - a) + tp);
			vec2 p5 = ((mix(tex(useModelTexture, u4), tex(useModelTexture, ub4), ta) - a) + tp);
			vec2 p6 = ((mix(tex(useModelTexture, u5), tex(useModelTexture, ub5), ta) - a) + tp);
			vec2 p7 = ((mix(tex(useModelTexture, u6), tex(useModelTexture, ub6), ta) - a) + tp);
			vec2 p8 = ((mix(tex(useModelTexture, u8), tex(useModelTexture, ub8), ta) - a) + tp);
			vec2 p9 = ((mix(tex(useModelTexture, u9), tex(useModelTexture, ub9), ta) - a) + tp);
			vec2 p10 = ((mix(tex(useModelTexture, u10), tex(useModelTexture, ub10), ta) - a) + tp);
			vec2 p11 = ((mix(tex(useModelTexture, u11), tex(useModelTexture, ub11), ta) - a) + tp);
			float ts = 0.0003/clamp(map(s1,0.3,1.,1.,2.),0.,1.);
      float ts2 = 0.00005/clamp(map(s1,0.3,1.,1.,2.),0.,1.)*0.;
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

      float trp = 2.;
      vec4 rpo  = rd4(ti*trp);
      vec4 rpo2 = rd4(ti*trp+78.45);
      vec4 rpo3 = rd4(ti*trp+425.36);
      float tt7 = 0.;
      for(int  i = 1 ; i <= 10 ; i++){
        float tti = float(i);
        tt7 += length(tex(useModelTexture, tc(ti2+1.-tti, _resx2, 7., itn, it))-tex(useModelTexture, tc(ti2-tti, _resx2, 7., itn, it)))/tti;
      }

      float ligne = pow(tt7*10.,1.5);
      float zp = mix(1.,0.25,s1);
      vec2 pp1 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo.x)),step(0.85,rpo.x)),step(0.77,rpo.x)),step(0.69,rpo.x)),step(0.62,rpo.x)),step(0.53,rpo.x)),step(0.46,rpo.x)),step(0.38,rpo.x))
      ,step(0.3,rpo.x)),step(0.23,rpo.x)),step(0.15,rpo.x)),step(0.08,rpo.x));
      vec2 pp2 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo.y)),step(0.85,rpo.y)),step(0.77,rpo.y)),step(0.69,rpo.y)),step(0.62,rpo.y)),step(0.53,rpo.y)),step(0.46,rpo.y)),step(0.38,rpo.y))
      ,step(0.3,rpo.y)),step(0.23,rpo.y)),step(0.15,rpo.y)),step(0.08,rpo.y));
      vec2 pp3 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo.z)),step(0.85,rpo.z)),step(0.77,rpo.z)),step(0.69,rpo.z)),step(0.62,rpo.z)),step(0.53,rpo.z)),step(0.46,rpo.z)),step(0.38,rpo.z))
      ,step(0.3,rpo.z)),step(0.23,rpo.z)),step(0.15,rpo.z)),step(0.08,rpo.z));
      vec2 pp4 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo.w)),step(0.85,rpo.w)),step(0.77,rpo.w)),step(0.69,rpo.w)),step(0.62,rpo.w)),step(0.53,rpo.w)),step(0.46,rpo.w)),step(0.38,rpo.w))
      ,step(0.3,rpo.w)),step(0.23,rpo.w)),step(0.15,rpo.w)),step(0.08,rpo.w));
      vec2 pp5 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo2.x)),step(0.85,rpo2.x)),step(0.77,rpo2.x)),step(0.69,rpo2.x)),step(0.62,rpo2.x)),step(0.53,rpo2.x)),step(0.46,rpo2.x)),step(0.38,rpo2.x))
      ,step(0.3,rpo2.x)),step(0.23,rpo2.x)),step(0.15,rpo2.x)),step(0.08,rpo2.x));
      vec2 pp6 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo2.y)),step(0.85,rpo2.y)),step(0.77,rpo2.y)),step(0.69,rpo2.y)),step(0.62,rpo2.y)),step(0.53,rpo2.y)),step(0.46,rpo2.y)),step(0.38,rpo2.y))
      ,step(0.3,rpo2.y)),step(0.23,rpo2.y)),step(0.15,rpo2.y)),step(0.08,rpo2.y));
      vec2 pp7 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo2.z)),step(0.85,rpo2.z)),step(0.77,rpo2.z)),step(0.69,rpo2.z)),step(0.62,rpo2.z)),step(0.53,rpo2.z)),step(0.46,rpo2.z)),step(0.38,rpo2.z))
      ,step(0.3,rpo2.z)),step(0.23,rpo2.z)),step(0.15,rpo2.z)),step(0.08,rpo2.z));
      vec2 pp8 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo2.w)),step(0.85,rpo2.w)),step(0.77,rpo2.w)),step(0.69,rpo2.w)),step(0.62,rpo2.w)),step(0.53,rpo2.w)),step(0.46,rpo2.w)),step(0.38,rpo2.w))
      ,step(0.3,rpo2.w)),step(0.23,rpo2.w)),step(0.15,rpo2.w)),step(0.08,rpo2.w));
      vec2 pp9 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo3.x)),step(0.85,rpo3.x)),step(0.77,rpo3.x)),step(0.69,rpo3.x)),step(0.62,rpo3.x)),step(0.53,rpo3.x)),step(0.46,rpo3.x)),step(0.38,rpo3.x))
      ,step(0.3,rpo3.x)),step(0.23,rpo3.x)),step(0.15,rpo3.x)),step(0.08,rpo3.x));
      vec2 pp10 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo3.y)),step(0.85,rpo3.y)),step(0.77,rpo3.y)),step(0.69,rpo3.y)),step(0.62,rpo3.y)),step(0.53,rpo3.y)),step(0.46,rpo3.y)),step(0.38,rpo3.y))
      ,step(0.3,rpo3.y)),step(0.23,rpo3.y)),step(0.15,rpo3.y)),step(0.08,rpo3.y));
      vec2 pp11 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo3.z)),step(0.85,rpo3.z)),step(0.77,rpo3.z)),step(0.69,rpo3.z)),step(0.62,rpo3.z)),step(0.53,rpo3.z)),step(0.46,rpo3.z)),step(0.38,rpo3.z))
      ,step(0.3,rpo3.z)),step(0.23,rpo3.z)),step(0.15,rpo3.z)),step(0.08,rpo3.z));
      vec2 pp12 = mix(tp,mix(p1,mix(p2,mix(p3,mix(p4,mix(p5,mix(p6,mix(p7,mix(p8,mix(p9,mix(p10,mix(p11,tp,
      step(0.92,rpo3.w)),step(0.85,rpo3.w)),step(0.77,rpo3.w)),step(0.69,rpo3.w)),step(0.62,rpo3.w)),step(0.53,rpo3.w)),step(0.46,rpo3.w)),step(0.38,rpo3.w))
      ,step(0.3,rpo3.w)),step(0.23,rpo3.w)),step(0.15,rpo3.w)),step(0.08,rpo3.w));
      float po1 = 0.; float po2 = 0.;float po3 = 0.; float po4 =0.;
      vec2 fac = vec2(1.,9./16.);
      if(ligne>0.2){
     po1 =smoothstep(0.0005*zp,0.,min(sdl(uv,pp1,pp2),sdl(uv,pp2,pp3)));
    po1 += smoothstep(0.00075*zp,0.,bez(uv,pp1,pp2,pp3));
    po1 += smoothstep(0.001*zp,0.,min(min(distance(uv*fac,pp1*fac),distance(uv*fac,pp2*fac)),
    distance(uv,pp3)));}
    if (ligne>0.4){
   po2 =smoothstep(0.0005*zp,0.,min(sdl(uv,pp4,pp5),sdl(uv,pp5,pp6)));
  po2 += smoothstep(0.00075*zp,0.,bez(uv,pp4,pp5,pp6));
  po2 += smoothstep(0.001*zp,0.,min(min(distance(uv*fac,pp4*fac),distance(uv*fac,pp5*fac)),
  distance(uv,pp6)));}
  if(ligne>0.6){
   po3 =smoothstep(0.0005*zp,0.,min(sdl(uv,pp7,pp8),sdl(uv,pp8,pp9)));
  po3 += smoothstep(0.00075*zp,0.,bez(uv,pp7,pp8,pp9));
  po3 += smoothstep(0.001*zp,0.,min(min(distance(uv*fac,pp7*fac),distance(uv*fac,pp8*fac)),
  distance(uv,pp9)));}
  if(ligne>0.8){
   po4 =smoothstep(0.0005*zp,0.,min(sdl(uv,pp10,pp11),sdl(uv,pp11,pp12)));
  po4 += smoothstep(0.00075*zp,0.,bez(uv,pp10,pp11,pp12));
  po4 += smoothstep(0.002*zp,0.,min(min(distance(uv*fac,pp10*fac),distance(uv*fac,pp11*fac)),
  distance(uv,pp12)));}
    float  pof = po1+po2+po3+po4;

    vec2 pr0s[8];pr0s[0] = p2;pr0s[1] = p4;pr0s[2] = p6;pr0s[3] = tp;pr0s[4] = p7;pr0s[5] = p5;pr0s[6] = p3;pr0s[7] = p1;
    vec2 pr2s[5];pr2s[0] = p1;pr2s[1] = p6;pr2s[2] = p10;pr2s[3] = p11;pr2s[4] = p7;
  vec2 pr3s[11];pr3s[0] =p1;pr3s[1] = p2;pr3s[2] = p4;pr3s[3] = p6;pr3s[4] = p8;pr3s[5] = p10;pr3s[6] = p11;
    pr3s[7] =p9;pr3s[8] = p7;pr3s[9] = p5;pr3s[10] = p3;
    vec2 pr4s[3];pr4s[0] = p2;pr4s[1] = p4;pr4s[2] = p6;
    vec2 pr5s[3];pr5s[0] = p3;pr5s[1] = p5;pr5s[2] = p7;
    vec2 pr6s[3];pr6s[0] = tp;pr6s[1] = p8;pr6s[2] = p10;
    vec2 pr7s[3];pr7s[0] = tp;pr7s[1] = p9;pr7s[2] = p11;
    vec2 pr8s[4];pr8s[0] = p2;pr8s[1] = p8;pr8s[2] = p9;pr8s[3] = p3;
    vec2 pr9s[4];pr9s[0] = p2;pr9s[1] = p10;pr9s[2] = p11;pr9s[3] = p3;
    float ps1 = 0.;float ps2 = 0.;float ps3 = 0.;float ps4 =0.;float ps5 = 0.;float ps6 = 0.;float ps7= 0.;float ps8 = 0.; float ps9 = 0.;
           if(rpo.x>1.-ligne){ps1 = smoothstep(0.0005*zp,0.,length(spo8(pr0s,uv)));}
           if(rpo.y>1.-ligne){ps2 = smoothstep(0.0005*zp,0.,length(spo5(pr2s,uv)));}
           if(rpo.z>1.-ligne){ps3 = smoothstep(0.0005*zp,0.,length(spo11(pr3s,uv)));}
           if(rpo.w>1.-ligne){ps4 = smoothstep(0.0005*zp,0.,length(spo3(pr4s,uv)));}
           if(rpo2.x>1.-ligne){ps5 = smoothstep(0.0005*zp,0.,length(spo3(pr5s,uv)));}
           if(rpo2.y>1.-ligne){ps6 = smoothstep(0.0005*zp,0.,length(spo3(pr6s,uv)));}
           if(rpo2.z>1.-ligne){ps7 = smoothstep(0.0005*zp,0.,length(spo3(pr7s,uv)));}
           if(rpo2.w>1.-ligne){ps8 = smoothstep(0.0005*zp,0.,length(spo4(pr8s,uv)));}
           if(rpo3.x>1.-ligne){ps9 = smoothstep(0.0005*zp,0.,length(spo4(pr9s,uv)));}
    pof += ps1+ps2+ps3+ps4+ps5+ps6+ps7+ps8+ps9;

        gl_FragColor =  (mix(vec4(1.,1.,1.,1.),vec4(1.,0.5,0.5,1.),p))*(1.-pof*0.2)*(1.-c*0.1);
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
      vec3 diff = vec3(vec2(1.,aspectRatio),0.);
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

let fragesTexture = createTextureAsync('Capture/capture0000.png');
let fragesTexture2 = createTextureAsyncs('Capture/capture0000.png');

function loadTextureWithSliderValue(value) {
    let textureUrl = 'Capture/capture' + pad(value, 4) + '.png'; // Assuming your texture files are named as capture0000.png, capture0001.png, etc.

    // Load the texture asynchronously
    fragesTexture = createTextureAsync(textureUrl);
    fragesTexture2 = createTextureAsyncs(textureUrl);
}
function pad(number, length) {
    var str = '' + number;
    while (str.length < length) {
        str = '0' + str;
    }
    return str;
}
function updateSliderValue(value) {
    document.getElementById("slider5-value").textContent = value.toFixed(2);
    s5 = value;
    loadTextureWithSliderValue(s5); // Load texture based on the slider value
}

// Event listener for slider change
document.getElementById("slider5").addEventListener("input", function() {
    updateSliderValue(parseFloat(this.value));
});


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
    gl.uniform1f(displayMaterial.uniforms.ti, performance.now() / 1000-tiValue);
  gl.uniform2f(displayMaterial.uniforms.resolution, canvas.width , canvas.height);

         gl.activeTexture(gl.TEXTURE1);
         gl.bindTexture(gl.TEXTURE_2D, modelTexture);
         gl.uniform1i(displayMaterial.uniforms.uModelTexture, 1); // Use modelTexture

         gl.uniform1i(displayMaterial.uniforms.uTex, fragesTexture.attach(0)); // Use fragesTexture

  gl.uniform1f(displayMaterial.uniforms.s1,s1);
  gl.uniform1f(displayMaterial.uniforms.s2,s2);
  gl.uniform1f(displayMaterial.uniforms.s3,s3);
  gl.uniform1f(displayMaterial.uniforms.s4,s4);
  gl.uniform1f(displayMaterial.uniforms.s5,s5);
  gl.uniform1f(displayMaterial.uniforms.useModelTexture,useModelTexture);
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


  function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
  }

  function createTransformFeedbackProgram(gl, vertexShaderSource, fragmentShaderSource, varyings) {
    const program = gl.createProgram();
    gl.attachShader(program, createShader(gl, vertexShaderSource, gl.VERTEX_SHADER));
    gl.attachShader(program, createShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    gl.transformFeedbackVaryings(program, varyings, gl.SEPARATE_ATTRIBS);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }

  function createVbo(gl, array, usage) {
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, array, usage !== undefined ? usage : gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    return vbo;
  }

  const VERTEX_SHADER =
`#version 300 es

out vec2 o_sound;

uniform float u_blockOffset;
uniform float u_sampleRate;
uniform float u_s4;
uniform sampler2D u_modelTexture1;
uniform sampler2D u_modelTexture;

uniform float useModelTexture;
vec2 tc(float _ti, float _resx2, float v, float itn, float it) {
return vec2(fract(_ti / _resx2), 1.-(fract(_ti / _resx2 / itn) / it+v/it));
}
float hash(float x){return fract(sin(x) * 897612.531);}
float voc(float t, float f, vec3 ft){float x = fract(t * f) / f;
float a=(sin(x*6.5*ft.x)*.4+sin(x*13.*ft.x)+sin(x*24.*ft.x)*.2);
float e=(sin(x*4.*ft.x)*.4+sin(x*22.*ft.x)+sin(x*25.*ft.x)*.2);
float o=(sin(x*5.*ft.x)*.4+sin(x*10.*ft.x)+sin(x*25.*ft.x)*.2);
float r = mix(mix(e,o,ft.y),a,smoothstep(0.5,1.,ft.z));
   return r* min(x * 1000., 1.) * exp(x * -200.);}
vec2 form(float t, vec3 var){
    vec2 v = vec2(0., 0.);
    for(int i = 0; i < 16; ++i){
        float h = float(i);
       	float m = voc(t + h / 3., 60. + pow(2.01, (h - 8.) * .2), var);
        float pan = hash(h);
        v.x += m * pan;
        v.y += m * (1. - pan);
    }
    return v*0.1 ;
   }
   vec2 she (float t,float n,float o){
   	vec2 v = vec2(0.);
     vec2 T = vec2(n,n+0.01);
   vec2  O = vec2(o+0.001,o);
   float ni = 56.;
     for (int i=0; i<=int(ni); i++) {
         float e = (1.-cos(6.28318530718*float(i)/ni))/2.;
         vec2 phase =floor(t/T) + pow(O,fract(t/T))-1.;
         phase = phase * pow(O,vec2(float(i)))*T/log(O) + float(i);
         v += e*sin(6.28318530718*phase);
     }
   v/=ni;return v;
 }
   vec2 tex(float m, vec2 uv ){ return texture(u_modelTexture, uv).xy;}
 vec2 crea(float ti,float _resx2,float itn,float it,float time,float ti2){

   float dp = smoothstep(0.,0.3,length(tex(useModelTexture,  tc(ti2, _resx2, 11., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 10., itn, it))));
   float dm = smoothstep(0.,0.3,length(tex(useModelTexture,  tc(ti2, _resx2, 6., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 5., itn, it))));
   float t0 = length(tex(useModelTexture,  tc(ti2+1., _resx2, 0., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 0., itn, it)));
   t0 += length(tex(useModelTexture,  tc(ti2+1., _resx2, 0., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 0., itn, it)));
   t0 += length(tex(useModelTexture,  tc(ti2+1., _resx2, 1., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 1., itn, it)));
   t0 += length(tex(useModelTexture,  tc(ti2+1., _resx2, 2., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 3., itn, it)));
   t0 += length(tex(useModelTexture,  tc(ti2+1., _resx2, 4., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 4., itn, it)));
   t0 += length(tex(useModelTexture,  tc(ti2+1., _resx2, 8., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 8., itn, it)));
   t0 += length(tex(useModelTexture,  tc(ti2+1., _resx2, 9., itn, it))-tex(useModelTexture,  tc(ti2, _resx2, 9., itn, it)));
   float tt7 = 0.;
   for(int  i = 1 ; i <= 10 ; i++){
     float tti = float(i);
     tt7 += length(tex(useModelTexture,  tc(ti2+1.-tti, _resx2, 7., itn, it))-tex(useModelTexture,  tc(ti2-tti, _resx2, 7., itn, it)))/tti;
   }
   vec2 s1 = vec2(0.);
   for(int j = 0 ; j < 4 ; j++){
     float ji = float(j);
     float jii = ji+1.;
     float t7 = mix(length(tex(useModelTexture,  tc(ti2+1.+ji*100., _resx2, 7., itn, it))-tex(useModelTexture,  tc(ti2+ji*100., _resx2, 7., itn, it))),
     length(tex(useModelTexture,  tc(ti2+2.+ji*100., _resx2, 7., itn, it))-tex(useModelTexture,   tc(ti2+1.+ji*100., _resx2, 7., itn, it))),
     smoothstep(0.,1.,fract(ti+ji*100.)));
   s1 += form(time*2./jii,2000.*vec3(t7,dp,dm))/jii;
 }//s1 *= 0.2;
   s1 += she(time,0.1+floor(pow(t0*100.,2.)),1.25)*tt7*2.;
   return normalize(s1);
 }
void main(void) {
  float time = u_blockOffset + float(gl_VertexID) / u_sampleRate;
  float ti = time *30.*u_s4;
  float ti2 = floor(ti);
  float _resx2 = 64.;
  float itn = floor(_resx2 / 12.);
  float it = _resx2 / itn;

  vec2 s1 = crea(ti,_resx2,itn,it,time,ti2);

  o_sound = s1*clamp(ti,0.,1.);
}
`;

  const FRAGMENT_SHADER =
`#version 300 es
void main(void) {}
`

  function createAudio() {
    const DURATION = 180; // seconds
    const SAMPLES = 65536;

    const audioCtx = new AudioContext();
    const audioBuffer = audioCtx.createBuffer(2, audioCtx.sampleRate * DURATION, audioCtx.sampleRate);


    const program = createTransformFeedbackProgram(gl2, VERTEX_SHADER, FRAGMENT_SHADER, ['o_sound']);
    const uniforms = getUniformLocations(gl2, program, ['u_sampleRate', 'u_blockOffset','u_s4','u_Texture1','u_Texture','useModelTexture']);

    const array = new Float32Array(2 * SAMPLES);
    const vbo = createVbo(gl2, array, gl2.DYNAMIC_COPY);
    const transformFeedback = gl2.createTransformFeedback();

    const numBlocks = (audioCtx.sampleRate * DURATION) / SAMPLES;
    const outputL = audioBuffer.getChannelData(0);
    const outputR = audioBuffer.getChannelData(1);

    gl2.bindTransformFeedback(gl2.TRANSFORM_FEEDBACK, transformFeedback);
    gl2.enable(gl2.RASTERIZER_DISCARD);
    gl2.useProgram(program);
    if(useModelTexture == false){
    gl2.activeTexture(gl2.TEXTURE0);
        gl2.bindTexture(gl2.TEXTURE_2D, modelTextures);
        gl2.uniform1i(uniforms['u_Texture0'], 0);
}

else{
    gl2.uniform1i(uniforms['u_Texture0'],fragesTexture2);}
    gl2.bindTransformFeedback(gl2.TRANSFORM_FEEDBACK, transformFeedback);
    gl2.enable(gl2.RASTERIZER_DISCARD);
    gl2.useProgram(program);
    gl2.uniform1f(uniforms['useModelTexture'], useModelTexture);
    gl2.uniform1f(uniforms['u_sampleRate'], audioCtx.sampleRate);
      gl2.uniform1f(uniforms['u_s4'],s4);
    for (let i = 0; i < numBlocks; i++) {
        gl2.uniform1f(uniforms['u_blockOffset'], i * SAMPLES / audioCtx.sampleRate);
        gl2.bindBufferBase(gl2.TRANSFORM_FEEDBACK_BUFFER, 0, vbo);
        gl2.beginTransformFeedback(gl2.POINTS);
        gl2.drawArrays(gl2.POINTS, 0, SAMPLES);
        gl2.endTransformFeedback();
        gl2.getBufferSubData(gl2.TRANSFORM_FEEDBACK_BUFFER, 0, array);

      for (let j = 0; j < SAMPLES; j++) {
        outputL[i * SAMPLES + j] = array[j * 2];
        outputR[i * SAMPLES + j] = array[j * 2 + 1];
      }
    }

    const node = audioCtx.createBufferSource();
    node.connect(audioCtx.destination);
    node.buffer = audioBuffer;
    node.loop = false;
    return node;
  }

  function getRandomInt(max) {
    return Math.floor(Math.random() * max);
  }

  let video;
  let poseNet;
  let pose;
  let skeleton;
  let audioCreated = false;
  let audioNode = null; // Define audioNode in the outer scope

  addEventListener('click', async () => {
    // document.getElementById("inputTexture").src='Capture/capture0149.png'
    // console.log(pose.keypoints)
      if (tecreate > 0) {
          if (audioNode) {
              // If audio node exists, stop and destroy it
              audioNode.stop();
              audioNode.disconnect();
          }

          // Capture the initial time when the click event happens
          tiValue = performance.now() / 1000;
          audioNode = createAudio(); // Create new audio node
          audioNode.start(0); // Start the audio with the calculated offset
      } else {
          console.log('Texture is not generated yet.');
      }
  });
  // document.getElementById('myButton').addEventListener('click', function() {
  //   let textureUrl = 'Capture/capture' + pad(getRandomInt(381), 4) + '.png';
  //   document.getElementById("inputTexture").src=textureUrl
  //   useModelTexture = !useModelTexture;
  // });

async function loadModelsPart2() {
    // Load the encoder and decoder models
    const decoder = await tf.loadGraphModel('model_json/latent_dim_4/augmented_captures/horizontal_flip/250_epochs/decoder/model.json');
    const encoder = await tf.loadGraphModel('model_json/latent_dim_4/augmented_captures/horizontal_flip/250_epochs/encoder/model.json');

    // !! update here the latent dim
    const latentDim = 4;
    const numExamplesToGenerate = 1; // Change the desired number of examples
    const generatedImages = [];

    const loaded_model = new CVAE(latentDim, encoder, decoder);

    for (let i = 0; i < numExamplesToGenerate; i++) {
      const randomVectorForGeneration = tf.randomNormal([1, latentDim]); // Generate a new random vector for each example
      const sample = await loaded_model.sample(randomVectorForGeneration);
      console.log("a");
      const generatedImage = await generateImage(loaded_model, sample);
      console.log(generatedImage);
      generatedImages.push(generatedImage);
    }
    // Display the generated images
    const imageContainer = document.getElementById('imageContainer');
    generatedImages.slice(1);
    generatedImages.forEach((imageData, index) => {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        const context = canvas.getContext('2d');
        context.putImageData(imageData, 0, 0);

        const img = document.createElement('img');
        img.src = canvas.toDataURL();
        img.alt = `Generated Image ${index + 1}`;
        img.classList.add('generatedImage');
        // console.log(img);
        imageContainer.appendChild(img);
    });
}

async function generateImage(model, sample) {
    const [mean, logvar] = await model.encode(sample);
    const z = model.reparameterize(mean, logvar);
    const prediction = await model.sample(z);
    // Convert the tensor to pixels
    const pixelsTensor = prediction.squeeze();
    const pixels = await tf.browser.toPixels(pixelsTensor);

    // Create an ImageData object from the pixel data
    const image = new ImageData(new Uint8ClampedArray(pixels), pixelsTensor.shape[1], pixelsTensor.shape[0]);

    return image;
}

// Utility function to convert ArrayBuffer to Base64
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

// loadModelsPart2();

function updateSelectedTexture(){
  loadModels()
  if (pose){
    console.log(pose.score);
  }
  // let textureUrl = 'Capture/capture' + pad(getRandomInt(381), 4) + '.png';
  // document.getElementById("inputTexture").src=textureUrl
}

setInterval(updateSelectedTexture, 4000);
