import {GaussianViewer} from './renderer.js';
const $=id=>document.getElementById(id);
let viewer,selectedFiles=[],previewURLs=[],activeJob=null,currentScene=null,health=null,sceneRequest=0,pollTimer=null;
const number=n=>new Intl.NumberFormat().format(n);
function showError(message,jobId=null){$('errorText').textContent=message;$('errorPanel').hidden=false;$('logLink').hidden=!jobId;if(jobId)$('logLink').href=`/api/jobs/${jobId}/files/worker.log`;}
function clearError(){$('errorPanel').hidden=true;}
async function request(url,options={}){
 const response=await fetch(url,{...options,signal:options.signal||AbortSignal.timeout(60000)});
 if(!response.ok){let body;try{body=await response.json();}catch{body={};}throw Error(typeof body.detail==='string'?body.detail:`Request failed (${response.status}). Check the local server window.`);}
 return response;
}
function setBusy(busy){
 for(const id of ['imageInput','engine','device','resolution','depthStrength','viewLimit'])$(id).disabled=busy;
 $('depthStrength').disabled=busy||$('engine').value==='anysplat';
 $('generateBtn').disabled=busy||!selectedFiles.length;$('generateBtn').textContent=busy?'Reconstructing…':'Create 3D scene ↗';$('progressPanel').hidden=!busy;$('cancelBtn').disabled=false;
}
function renderPreviews(){
 const container=$('imagePreviews');container.replaceChildren();container.hidden=!selectedFiles.length;$('uploadPrompt').hidden=!!selectedFiles.length;
 selectedFiles.forEach((item,index)=>{
  const card=document.createElement('div');card.className='image-card';
  const img=document.createElement('img');img.src=item.url;img.alt=`Source view ${index+1}: ${item.file.name}`;
  const remove=document.createElement('button');remove.type='button';remove.className='remove-image';remove.textContent='×';remove.setAttribute('aria-label',`Remove ${item.file.name}`);
  remove.addEventListener('click',event=>{event.stopPropagation();URL.revokeObjectURL(item.url);selectedFiles.splice(index,1);renderPreviews();updateSelection();});
  const order=document.createElement('span');order.className='view-order';order.textContent=index+1;
  card.append(img,order,remove);container.append(card);
 });
}
function updateSelection(){
 const total=selectedFiles.reduce((sum,item)=>sum+item.file.size,0);
 $('filename').textContent=selectedFiles.length?`${selectedFiles.length} view${selectedFiles.length===1?'':'s'} selected · ${(total/1024/1024).toFixed(2)} MB total`:'Move around the same subject. Keep neighboring photographs overlapping.';
 $('generateBtn').disabled=!selectedFiles.length;
}
function chooseFiles(files){
 if(activeJob)return;clearError();const incoming=[...files];
 if(incoming.length>16){showError('Choose at most sixteen photographs.');return;}
 for(const file of incoming){
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){showError('Choose only JPEG, PNG or WebP photographs.');return;}
  if(file.size>20*1024*1024){showError(`${file.name} is over 20 MB. Save a smaller copy and try again.`);return;}
 }
 previewURLs.forEach(url=>URL.revokeObjectURL(url));
 previewURLs=incoming.map(file=>URL.createObjectURL(file));
 selectedFiles=incoming.map((file,index)=>({file,url:previewURLs[index]}));
 renderPreviews();updateSelection();
}
$('imageInput').addEventListener('change',e=>chooseFiles(e.target.files));
for(const type of ['dragenter','dragover'])$('dropZone').addEventListener(type,e=>{e.preventDefault();$('dropZone').classList.add('drag-over');});
for(const type of ['dragleave','drop'])$('dropZone').addEventListener(type,e=>{e.preventDefault();$('dropZone').classList.remove('drag-over');});
$('dropZone').addEventListener('drop',e=>chooseFiles(e.dataTransfer.files));
$('engine').addEventListener('change',()=>{
 const anysplat=$('engine').value==='anysplat';$('qualityField').hidden=anysplat;$('viewBudgetField').hidden=!anysplat;$('depthStrength').disabled=anysplat;
 $('device').querySelector('option[value="cpu"]').disabled=anysplat;
 if(anysplat&&$('device').value==='cpu')$('device').value='auto';
 $('engineDescription').textContent=anysplat?'Jointly predicts cameras and 3D Gaussians from two or more uncalibrated views. NVIDIA CUDA required.':'Predicts depth from one image, then builds a visible Gaussian surface. CPU and CUDA supported.';
});
$('engine').dispatchEvent(new Event('change'));
$('depthStrength').addEventListener('input',()=>{$('depthValue').textContent=Number($('depthStrength').value).toFixed(2)+'×';});

async function loadScene(id=null){
 if(!viewer)return;
 const token=++sceneRequest,base=id?`/api/jobs/${id}/files`:'/api/demo';$('viewerLoading').hidden=false;
 try{
  const [metaResponse,sceneResponse]=await Promise.all([request(`${base}/scene.json`),request(`${base}/scene.gsb`)]);
  const meta=await metaResponse.json(),buffer=await sceneResponse.arrayBuffer();if(token!==sceneRequest)return;
  viewer.load(buffer,meta);currentScene=id;
  $('sceneTitle').textContent=id?'Reconstructed scene':'Renderer calibration';
  const isAnySplat=meta.method==='anysplat';
  $('sceneBadge').textContent=meta.method==='demo'?'SYNTHETIC DEMO':isAnySplat?'ANYSPLAT · MULTI-VIEW':'DEPTH RECONSTRUCTION';
  $('engineStat').textContent=meta.method==='demo'?'Procedural demo':isAnySplat?`AnySplat · ${meta.input_count} views`:'Depth Anything V2 Small';
  $('timeStat').textContent=meta.seconds?`${meta.seconds.toFixed(1)} s · ${meta.device.toUpperCase()}`:'—';
  $('splatCount').textContent=number(meta.preview_gaussians);
  $('splatCount').title=`${number(meta.gaussians)} Gaussians in the full PLY export`;
  $('sceneNote').textContent=meta.method==='demo'?'This 3D calibration sculpture lets you test the viewer without a model. Upload a photograph to create your own scene.':meta.limitation+(meta.gaussians>meta.preview_gaussians?` The interactive preview uses ${number(meta.preview_gaussians)} of ${number(meta.gaussians)} Gaussians; the PLY keeps the full result.`:'');
  $('downloadPly').href=`${base}/scene.ply`;$('downloadPly').download=id?'gaussian-scene.ply':'calibration.ply';
  $('depthLink').hidden=meta.method!=='depth';$('depthLink').href=`${base}/depth.png`;$('metadataLink').hidden=!id;$('metadataLink').href=`${base}/scene.json`;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view==='front'));
  if(id){const job=await (await request(`/api/jobs/${id}`)).json();if(token===sceneRequest)$('sceneTitle').textContent=job.name;}
 }catch(e){if(token===sceneRequest)showError(`Could not load the 3D scene: ${e.message}`);}
 finally{if(token===sceneRequest)$('viewerLoading').hidden=true;}
}

async function refreshHealth(){
 try{
  health=await (await request('/api/health')).json();const h=health.hardware;
  $('hardwareTitle').textContent=h.status==='checking'?'Checking this computer…':h.cuda?(h.gpu||'NVIDIA CUDA ready'):'CPU inference available';
  $('hardwareDetail').textContent=h.status==='checking'?'Detecting the compute environment':h.cuda?`${h.vram_gb} GB VRAM · PyTorch CUDA ${h.torch_cuda_runtime}`:`${h.ram_gb||'?'} GB RAM · local WebGL viewer`;
  if(h.error){$('hardwareTitle').textContent='Environment needs setup';$('hardwareDetail').textContent='Run setup.ps1, then restart the app.';}
  if(h.status==='checking')setTimeout(refreshHealth,3000);
  if(health.active_job&&!activeJob){activeJob=health.active_job;setBusy(true);pollJob();}
 }catch(e){$('hardwareTitle').textContent='Server disconnected';$('hardwareDetail').textContent='Keep the app’s terminal open.';}
}
$('refreshHardware').addEventListener('click',refreshHealth);

async function refreshHistory(){
 try{
  const jobs=await (await request('/api/jobs')).json();$('historyCount').textContent=jobs.length;const list=$('historyList');list.replaceChildren();
  if(!jobs.length){const p=document.createElement('p');p.className='field-note';p.textContent='Your reconstructions will appear here.';list.append(p);}
  for(const job of jobs){
   const button=document.createElement('button');button.className='history-item';button.type='button';const img=document.createElement('img');img.src=`/api/jobs/${job.id}/files/thumbnail.jpg`;img.alt='';img.loading='lazy';
   const text=document.createElement('div'),title=document.createElement('strong'),sub=document.createElement('span');title.textContent=job.name;sub.textContent=`${job.status} · ${job.engine==='anysplat'?'AnySplat':'Depth Anything'}`;text.append(title,sub);button.append(img,text);
   button.addEventListener('click',()=>{clearError();if(job.status==='completed')loadScene(job.id);else if(job.status==='running'){activeJob=job.id;setBusy(true);pollJob();}else showError(job.message,job.id);});list.append(button);
  }
 }catch(e){showError(e.message);}
}

$('createForm').addEventListener('submit',async e=>{
 e.preventDefault();if(!selectedFiles.length||activeJob)return;clearError();const engine=$('engine').value;
 if(engine==='depth'&&selectedFiles.length!==1){showError('Depth Anything accepts one image. Choose AnySplat for multiple views.');return;}
 if(engine==='anysplat'&&selectedFiles.length<2){showError('AnySplat needs at least two overlapping images.');return;}
 if(health&&!health.models[engine]){showError(engine==='anysplat'?'AnySplat is not installed. Run Setup NVIDIA Workstation.cmd from the app folder, then click refresh.':'Download the depth model first: run .venv\\Scripts\\python.exe scripts\\download_models.py --model depth from the app folder, then click refresh.');return;}
 if(engine==='anysplat'&&health?.hardware.status!=='checking'&&!health?.hardware.cuda){showError('AnySplat requires NVIDIA CUDA. Run Setup NVIDIA Workstation.cmd on the office workstation.');return;}
 if($('device').value==='cuda'&&health?.hardware.status!=='checking'&&!health?.hardware.cuda){showError('CUDA is not available in this Python environment. Select CPU or Auto for Depth Anything.');return;}
 const data=new FormData();for(const item of selectedFiles)data.append('images',item.file,item.file.name);data.append('engine',engine);data.append('device',$('device').value);data.append('resolution',$('resolution').value);data.append('depth_strength',$('depthStrength').value);data.append('view_limit',$('viewLimit').value);
 setBusy(true);$('progressBar').value=0;$('progressPercent').textContent='0%';$('progressMessage').textContent='Uploading to your local Python server…';
 try{const job=await (await request('/api/jobs',{method:'POST',body:data})).json();activeJob=job.id;await refreshHistory();pollJob();}
 catch(error){showError(error.message);setBusy(false);}
});

async function pollJob(){
 clearTimeout(pollTimer);const id=activeJob;if(!id)return;
 try{
  const job=await (await request(`/api/jobs/${id}`)).json();if(activeJob!==id)return;
  $('progressBar').value=job.progress;$('progressPercent').textContent=`${job.progress}%`;$('progressMessage').textContent=job.message;
  if(job.status==='running'||job.status==='queued'){pollTimer=setTimeout(pollJob,1200);return;}
  activeJob=null;setBusy(false);await refreshHistory();
  if(job.status==='completed'){await loadScene(id);if(matchMedia('(max-width:740px)').matches)$('viewerWrap').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth',block:'center'});}
  else if(job.status==='failed')showError(job.message,id);
 }catch(e){$('progressMessage').textContent='Connection interrupted. Reconnecting to the local job…';pollTimer=setTimeout(pollJob,3000);}
}
$('cancelBtn').addEventListener('click',async()=>{if(!activeJob)return;$('cancelBtn').disabled=true;try{await request(`/api/jobs/${activeJob}/cancel`,{method:'POST'});pollJob();}catch(e){showError(e.message);$('cancelBtn').disabled=false;}});

function cameraPreset(name){viewer?.preset(name);document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('selected',b.dataset.view===name));}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>cameraPreset(b.dataset.view)));
$('resetBtn').addEventListener('click',()=>cameraPreset('front'));
$('zoomIn').addEventListener('click',()=>viewer?.zoom(.8));$('zoomOut').addEventListener('click',()=>viewer?.zoom(1.25));
$('orbitBtn').addEventListener('click',()=>{if(viewer){viewer.auto=!viewer.auto;$('orbitBtn').setAttribute('aria-pressed',viewer.auto);}});
$('splatSize').addEventListener('input',()=>{if(viewer){viewer.scale=Number($('splatSize').value);viewer.dirty=true;}});
$('snapshotBtn').addEventListener('click',async()=>{try{if(!viewer?.count)throw Error('Load a scene first.');const blob=await viewer.screenshot(),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='gaussian-view.png';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);}catch(e){showError(e.message);}});
$('fullscreenBtn').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('viewerWrap').requestFullscreen();}catch(e){showError('Fullscreen is unavailable in this browser window.');}});
$('demoBtn').addEventListener('click',()=>{clearError();loadScene();});
$('helpBtn').addEventListener('click',()=>$('helpDialog').showModal());$('closeHelp').addEventListener('click',()=>$('helpDialog').close());
try{viewer=new GaussianViewer($('viewer'),stats=>{$('viewer').dataset.gaussians=stats.count;$('viewer').dataset.camera=JSON.stringify([stats.yaw,stats.pitch]);},showError);loadScene();}
catch(e){showError(e.message);$('renderState').textContent='RENDERER UNAVAILABLE';}
refreshHealth();refreshHistory();
