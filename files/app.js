const input=document.querySelector('#fileInput'),preview=document.querySelector('#preview'),empty=document.querySelector('#emptyState'),detectBtn=document.querySelector('#detectBtn'),clearBtn=document.querySelector('#clearBtn'),result=document.querySelector('#result');let selectedFile=null,stream=null;
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}

function placeholderHTML(){return '<div class="result-placeholder"><div class="mini-card"><span>?</span><i>✦</i></div><b>Your card will show up here</b><span>We’ll only name it when the picture is clear enough.</span></div>'}

function say(message,found=false,stats=null,visuals=null){
  let html=`<div class="result-ready"><div class="status-tag ${found?'':'warn'}">${found?'CARD FOUND':'NEEDS A CLOSER LOOK'}</div><h3>${found?'Let’s try again':'Let’s try again'}</h3><p>${esc(message)}</p>`;
  html+=visualsHTML(visuals)+statsHTML(stats);
  html+='</div>';
  result.innerHTML=html;
}

function visualsHTML(visuals){
  if(!visuals)return'';
  const items=[['contour','Contour'],['edges','Edges'],['warped','Straightened'],['activation_heatmap','Feature map']];
  const tiles=items.filter(([k])=>visuals[k]).map(([k,label])=>`<figure><img src="${visuals[k]}" alt="${label}"><figcaption>${label}</figcaption></figure>`).join('');
  return tiles?`<div class="visual-gallery">${tiles}</div>`:'';
}

function statsHTML(stats){
  if(!stats)return'';
  const rows=[['Sharpness',stats.sharpness],['Brightness',stats.mean_brightness],['Brightness σ',stats.brightness_std],['Edge density',stats.edge_density],['Aspect ratio',stats.aspect_ratio]];
  const cells=rows.map(([label,val])=>`<div class="stat-cell"><span>${label}</span><b>${val}</b></div>`).join('');
  return `<div class="stats-grid">${cells}</div>`;
}

function confidenceHTML(confidence,classProbabilities){
  if(confidence==null)return'';
  let bars='';
  if(classProbabilities){
    const sorted=Object.entries(classProbabilities).sort((a,b)=>b[1]-a[1]);
    bars=sorted.map(([label,pct])=>`<div class="prob-row"><span>${esc(label)}</span><div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div><b>${pct}%</b></div>`).join('');
  }
  return `<div class="confidence-block"><div class="confidence-headline"><span>Detection confidence</span><b>${confidence}%</b></div><div class="prob-list">${bars}</div></div>`;
}

function setFile(file){if(!file)return;if(!file.type.startsWith('image/')){say('Choose an image file to detect a card.');return}selectedFile=file;preview.src=URL.createObjectURL(file);preview.hidden=false;empty.hidden=true;detectBtn.disabled=false;clearBtn.hidden=false}
document.querySelector('#chooseBtn').onclick=()=>input.click();input.onchange=e=>setFile(e.target.files[0]);const dz=document.querySelector('#dropzone');dz.ondragover=e=>{e.preventDefault();dz.style.borderColor='#80c755'};dz.ondragleave=()=>dz.style.borderColor='';dz.ondrop=e=>{e.preventDefault();dz.style.borderColor='';setFile(e.dataTransfer.files[0])};
clearBtn.onclick=()=>{selectedFile=null;input.value='';preview.hidden=true;empty.hidden=false;detectBtn.disabled=true;clearBtn.hidden=true;result.innerHTML=placeholderHTML()};

detectBtn.onclick=async()=>{
  if(!selectedFile)return;
  detectBtn.disabled=true;detectBtn.innerHTML='Looking closely…';
  try{
    const data=new FormData();data.append('file',selectedFile);
    const response=await fetch('/api/detect',{method:'POST',body:data}),p=await response.json();
    if(!response.ok)throw Error(p.detail||'Something went wrong.');
    if(p.status==='identified'){
      const title=p.card_type;
      let html=`<div class="result-ready"><div class="status-tag">CARD FOUND</div><h3>${esc(title)}</h3><p>${esc(p.message)}</p>`;
      html+=confidenceHTML(p.confidence,p.class_probabilities);
      html+=visualsHTML(p.visuals);
      html+=statsHTML(p.stats);
      html+='</div>';
      result.innerHTML=html;
    }else{
      say(p.message,p.detected,p.stats,p.visuals);
    }
  }catch(e){say(e.message)}
  finally{detectBtn.disabled=false;detectBtn.innerHTML='Detect my card <span>→</span>'}
};

const modal=document.querySelector('#cameraModal'),video=document.querySelector('#video');document.querySelector('#cameraBtn').onclick=async()=>{try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});video.srcObject=stream;modal.hidden=false}catch{say('Camera access was blocked. Choose an image instead.')}};function stopCamera(){if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;modal.hidden=true}document.querySelector('#closeCamera').onclick=stopCamera;modal.onclick=e=>{if(e.target===modal)stopCamera()};document.querySelector('#captureBtn').onclick=()=>{const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext('2d').drawImage(video,0,0);canvas.toBlob(blob=>{setFile(new File([blob],'camera-capture.jpg',{type:'image/jpeg'}));stopCamera()},'image/jpeg',.92)};
