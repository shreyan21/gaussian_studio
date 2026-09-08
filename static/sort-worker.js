// O(N + 65536) quantized camera-depth counting sort, back-to-front.
let positions=null;
self.onmessage=({data})=>{
 if(data.positions){positions=data.positions;return;}
 if(!positions||!data.view)return;
 const begin=performance.now(),m=data.view,n=positions.length/3,depths=new Float32Array(n);let min=Infinity,max=-Infinity;
 for(let i=0;i<n;i++){const j=i*3,d=m[2]*positions[j]+m[6]*positions[j+1]+m[10]*positions[j+2]+m[14];depths[i]=d;min=Math.min(min,d);max=Math.max(max,d);}
 const bins=new Uint32Array(65536),keys=new Uint16Array(n),scale=65535/Math.max(max-min,1e-8);
 for(let i=0;i<n;i++){const k=Math.max(0,Math.min(65535,Math.floor((depths[i]-min)*scale)));keys[i]=k;bins[k]++;}
 let sum=0;for(let i=0;i<bins.length;i++){const count=bins[i];bins[i]=sum;sum+=count;}
 const indices=new Uint32Array(n);for(let i=0;i<n;i++)indices[bins[keys[i]]++]=i;
 self.postMessage({indices,serial:data.serial,ms:performance.now()-begin},[indices.buffer]);
};
